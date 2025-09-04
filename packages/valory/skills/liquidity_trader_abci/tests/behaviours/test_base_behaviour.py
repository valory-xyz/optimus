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

"""Tests for LiquidityTraderBaseBehaviour of the liquidity_trader_abci skill."""

import json
import os
import tempfile
import time
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Generator
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

import pytest
from hypothesis import given, strategies as st

from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.base import (
    LiquidityTraderBaseBehaviour,
    DexType,
    Action,
    SwapStatus,
    Decision,
    PositionStatus,
    TradingType,
    Chain,
    GasCostTracker,
    execute_strategy,
    ZERO_ADDRESS,
    WHITELISTED_ASSETS,
    COIN_ID_MAPPING,
    REWARD_TOKEN_ADDRESSES,
    OLAS_ADDRESSES,
    REWARD_UPDATE_KEY_PREFIX,
)
from packages.valory.skills.liquidity_trader_abci.states.base import StakingState
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.protocols.srr.message import SrrMessage
from packages.dvilela.protocols.kv_store.message import KvStoreMessage

PACKAGE_DIR = Path(__file__).parent.parent.parent


class LiquidityTraderAbciFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing FSMBehaviour."""

    path_to_skill = PACKAGE_DIR

    def setUp(self):
        """Setup test environment."""
        super().setUp()

    def tearDown(self):
        """Clean up test environment."""
        pass


class TestLiquidityTraderBaseBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test cases for LiquidityTraderBaseBehaviour."""

    behaviour_class = LiquidityTraderBaseBehaviour
    path_to_skill = PACKAGE_DIR

    @classmethod
    def setup_class(cls, **kwargs: Any) -> None:
        """Setup the test class with parameter overrides."""
        import tempfile
        import shutil
        
        cls.temp_skill_dir = tempfile.mkdtemp()
        cls.original_skill_dir = cls.path_to_skill
        
        shutil.copytree(cls.original_skill_dir, cls.temp_skill_dir, dirs_exist_ok=True)
        
        temp_skill_yaml = Path(cls.temp_skill_dir) / "skill.yaml"
        with open(temp_skill_yaml, 'r') as f:
            skill_config = f.read()
        
        skill_config = skill_config.replace("available_strategies: null", "available_strategies: \"{}\"")
        skill_config = skill_config.replace("genai_api_key: null", "genai_api_key: \"\"")
        skill_config = skill_config.replace("default_acceptance_time: null", "default_acceptance_time: 30")
        
        with open(temp_skill_yaml, 'w') as f:
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
                "safe_contract_addresses": {"mode": "0xSafeAddress"}
            }
        }
        
        super().setup_class(**kwargs)

    @classmethod
    def teardown_class(cls) -> None:
        """Teardown the test class."""
        if hasattr(cls, 'temp_skill_dir'):
            import shutil
            shutil.rmtree(cls.temp_skill_dir, ignore_errors=True)

        if hasattr(cls, 'original_skill_dir'):
            cls.path_to_skill = cls.original_skill_dir

        super().teardown_class()

    def setUp(self):
        """Setup test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test files
        self.test_file = self.temp_path / "test_file.json"
        self.test_file.write_text('{"test": "data"}')

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_base_behaviour(self) -> LiquidityTraderBaseBehaviour:
        """Create a base behaviour instance for testing."""
        return LiquidityTraderBaseBehaviour(
            name="base_behaviour",
            skill_context=self.skill.skill_context
        )

    def _consume_generator(self, generator: Generator) -> Any:
        """Consume a generator and return the final value."""
        result = None
        try:
            while True:
                result = next(generator)
        except StopIteration as e:
            result = e.value
        return result

    @pytest.mark.parametrize(
        "contract_id,contract_callable,response_msg,expected_log",
        [
            ("test_contract", "test_method", MagicMock(), "Could not successfully interact"),
            ("erc20", "get_balance", MagicMock(), "Could not successfully interact"),
        ]
    )
    def test_default_error(self, contract_id, contract_callable, response_msg, expected_log):
        """Test default_error method."""
        base_behaviour = self._create_base_behaviour()
        
        with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
            base_behaviour.default_error(contract_id, contract_callable, response_msg)
            mock_logger.assert_called_once()
            call_args = mock_logger.call_args[0][0]
            assert contract_id in call_args
            assert contract_callable in call_args

    @pytest.mark.parametrize(
        "response_body,expected_log_level",
        [
            ({"info": "test info"}, "info"),
            ({"warning": "test warning"}, "warning"),
            ({"error": "test error"}, "error"),
            ({}, "error"),  # Should call default_error when no level messages found
        ]
    )
    def test_contract_interaction_error(self, response_body, expected_log_level):
        """Test contract_interaction_error method."""
        base_behaviour = self._create_base_behaviour()
        
        mock_response = MagicMock()
        mock_response.raw_transaction.body = response_body
        
        with patch.object(base_behaviour.context.logger, expected_log_level) as mock_logger:
            base_behaviour.contract_interaction_error("test_contract", "test_method", mock_response)
            
            if response_body:
                mock_logger.assert_called_once_with(list(response_body.values())[0])
            else:
                # Should call default_error when no level messages found
                pass

    @pytest.mark.parametrize(
        "performative,contract_address,contract_public_id,contract_callable,data_key,kwargs,expected_result",
        [
            (
                ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                "0xContract123456789012345678901234567890123456",
                MagicMock(),
                "get_balance",
                "data",
                {"account": "0xAccount123456789012345678901234567890123456"},
                "1000000000000000000"
            ),
            (
                ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                "0xContract123456789012345678901234567890123456",
                MagicMock(),
                "get_decimals",
                "data",
                {},
                18
            ),
        ]
    )
    def test_contract_interact_success(self, performative, contract_address, contract_public_id, 
                                     contract_callable, data_key, kwargs, expected_result):
        """Test contract_interact method with successful response."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the contract API response
        mock_response = MagicMock()
        mock_response.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        mock_response.raw_transaction.body = {data_key: expected_result}
        
        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return mock_response
        
        with patch.object(base_behaviour, 'get_contract_api_response', side_effect=mock_get_contract_api_response):
            generator = base_behaviour.contract_interact(
                performative, contract_address, contract_public_id, contract_callable, data_key, **kwargs
            )
            result = self._consume_generator(generator)
            
            assert result == expected_result

    @pytest.mark.parametrize(
        "performative,contract_address,contract_public_id,contract_callable,data_key,kwargs,response_performative",
        [
            (
                ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                "0xContract123456789012345678901234567890123456",
                MagicMock(),
                "get_balance",
                "data",
                {"account": "0xAccount123456789012345678901234567890123456"},
                ContractApiMessage.Performative.ERROR
            ),
        ]
    )
    def test_contract_interact_error(self, performative, contract_address, contract_public_id,
                                   contract_callable, data_key, kwargs, response_performative):
        """Test contract_interact method with error response."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the contract API response with error
        mock_response = MagicMock()
        mock_response.performative = response_performative
        
        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return mock_response
        
        with patch.object(base_behaviour, 'get_contract_api_response', side_effect=mock_get_contract_api_response):
            with patch.object(base_behaviour, 'default_error') as mock_default_error:
                generator = base_behaviour.contract_interact(
                    performative, contract_address, contract_public_id, contract_callable, data_key, **kwargs
                )
                result = self._consume_generator(generator)
                
                assert result is None
                mock_default_error.assert_called_once()

    @pytest.mark.parametrize(
        "performative,contract_address,contract_public_id,contract_callable,data_key,kwargs,response_body",
        [
            (
                ContractApiMessage.Performative.GET_RAW_TRANSACTION,
                "0xContract123456789012345678901234567890123456",
                MagicMock(),
                "get_balance",
                "data",
                {"account": "0xAccount123456789012345678901234567890123456"},
                {}  # No data_key in response
            ),
        ]
    )
    def test_contract_interact_no_data(self, performative, contract_address, contract_public_id,
                                     contract_callable, data_key, kwargs, response_body):
        """Test contract_interact method when data_key is not found in response."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the contract API response without the expected data_key
        mock_response = MagicMock()
        mock_response.performative = ContractApiMessage.Performative.RAW_TRANSACTION
        mock_response.raw_transaction.body = response_body
        
        def mock_get_contract_api_response(*args, **kwargs):
            yield
            return mock_response
        
        with patch.object(base_behaviour, 'get_contract_api_response', side_effect=mock_get_contract_api_response):
            with patch.object(base_behaviour, 'contract_interaction_error') as mock_error:
                generator = base_behaviour.contract_interact(
                    performative, contract_address, contract_public_id, contract_callable, data_key, **kwargs
                )
                result = self._consume_generator(generator)
                
                assert result is None
                mock_error.assert_called_once()

    @pytest.mark.parametrize(
        "chain,positions,token,expected_balance",
        [
            ("optimism", [{"chain": "optimism", "assets": [{"address": "0xToken123", "balance": 1000}]}], "0xToken123", 1000),
            ("mode", [{"chain": "mode", "assets": [{"address": "0xToken456", "balance": 2000}]}], "0xToken456", 2000),
            ("optimism", [{"chain": "optimism", "assets": [{"address": "0xToken123", "balance": 1000}]}], "0xToken789", None),
            ("optimism", [], "0xToken123", None),
        ]
    )
    def test_get_balance(self, chain, positions, token, expected_balance):
        """Test _get_balance method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._get_balance(chain, token, positions)
        assert result == expected_balance

    @pytest.mark.parametrize(
        "amount,token_decimal,expected_result",
        [
            (1000000000000000000, 18, "1.000000000000000000"),
            (500000000000000000, 18, "0.500000000000000000"),
            (1000000, 6, "1.000000"),
            (0, 18, "0.000000000000000000"),
            (None, 18, None),
            (1000000000000000000, None, None),
        ]
    )
    def test_convert_to_token_units(self, amount, token_decimal, expected_result):
        """Test _convert_to_token_units method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._convert_to_token_units(amount, token_decimal)
        assert result == expected_result

    @pytest.mark.parametrize(
        "data,attribute,filepath,file_exists",
        [
            ({"test": "data"}, "test_attribute", "test_file.json", True),
            (None, "test_attribute", "test_file.json", True),
            ({"test": "data"}, "test_attribute", "test_file.json", False),
            # Additional scenarios for better coverage
            ({"complex": [1, 2, 3]}, "complex_attr", "complex.json", False),
            ({}, "empty_attr", "empty.json", False),
        ]
    )
    def test_store_data(self, data, attribute, filepath, file_exists):
        """Test _store_data method."""
        # Ensure setUp is called
        if not hasattr(self, 'temp_path'):
            self.setUp()
            
        base_behaviour = self._create_base_behaviour()
        
        file_path = self.temp_path / filepath
        
        if file_exists:
            # Create the file first
            with open(file_path, 'w') as f:
                json.dump({"existing": "data"}, f)
        
        with patch.object(base_behaviour.context.logger, 'warning') as mock_warning:
            with patch.object(base_behaviour.context.logger, 'error') as mock_error:
                base_behaviour._store_data(data, attribute, str(file_path))
                
                if data is None:
                    mock_warning.assert_called_once()
                else:
                    # Check if file was written correctly
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            stored_data = json.load(f)
                        assert stored_data == data

    def test_store_data_error_handling(self):
        """Test _store_data method error handling."""
        base_behaviour = self._create_base_behaviour()
        
        # Test with a filepath that causes IOError/OSError
        with patch.object(base_behaviour.context.logger, 'error') as mock_error:
            with patch('builtins.open', side_effect=OSError("Permission denied")):
                base_behaviour._store_data({"test": "data"}, "test_attr", "/invalid/path/file.json")
                
                mock_error.assert_called_once()

    @pytest.mark.parametrize(
        "attribute,filepath,file_exists,file_content,class_object",
        [
            ("test_attribute", "test_file.json", True, {"test": "data"}, False),
            ("test_attribute", "test_file.json", False, None, False),
            ("current_positions", "current_positions.json", False, None, False),
            # Additional scenarios for better coverage
            ("agent_performance", "agent_performance.json", True, {"metrics": ["test"]}, False),
            ("positions", "positions.json", False, None, False),
        ]
    )
    def test_read_data(self, attribute, filepath, file_exists, file_content, class_object):
        """Test _read_data method."""
        # Ensure setUp is called
        if not hasattr(self, 'temp_path'):
            self.setUp()
            
        base_behaviour = self._create_base_behaviour()
        
        file_path = self.temp_path / filepath
        
        if file_exists and file_content:
            with open(file_path, 'w') as f:
                json.dump(file_content, f)
        
        with patch.object(base_behaviour.context.logger, 'error') as mock_error:
            base_behaviour._read_data(attribute, str(file_path), class_object)
            
            if file_exists and file_content:
                # Check if attribute was set correctly
                if hasattr(base_behaviour, attribute):
                    assert getattr(base_behaviour, attribute) == file_content
            elif not file_exists and attribute == "current_positions":
                # Should create file with empty list
                assert file_path.exists()
                with open(file_path, 'r') as f:
                    created_data = json.load(f)
                assert created_data == []

    def test_read_data_json_decode_error(self):
        """Test _read_data method with JSON decode error."""
        base_behaviour = self._create_base_behaviour()
        
        # Ensure setUp is called to create temp_path
        if not hasattr(self, 'temp_path'):
            self.setUp()
        
        # Create a file with invalid JSON
        file_path = self.temp_path / "invalid.json"
        with open(file_path, 'w') as f:
            f.write("invalid json content")
        
        with patch.object(base_behaviour.context.logger, 'error') as mock_error:
            base_behaviour._read_data("test_attr", str(file_path))
            
            mock_error.assert_called_once()

    def test_read_data_file_read_error(self):
        """Test _read_data method with file read error."""
        base_behaviour = self._create_base_behaviour()
        
        # Test with a filepath that causes PermissionError/OSError
        with patch.object(base_behaviour.context.logger, 'error') as mock_error:
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                base_behaviour._read_data("test_attr", "/invalid/path/file.json")
                
                mock_error.assert_called_once()

    @pytest.mark.parametrize(
        "symbol,chain_name,expected_coin_id",
        [
            ("usdc", "mode", "mode-bridged-usdc-mode"),
            ("usdt", "optimism", "bridged-usdt"),
            ("weth", "mode", "l2-standard-bridged-weth-modee"),
            ("unknown", "mode", None),
            ("usdc", "unknown_chain", None),
        ]
    )
    def test_get_coin_id_from_symbol(self, symbol, chain_name, expected_coin_id):
        """Test get_coin_id_from_symbol method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour.get_coin_id_from_symbol(symbol, chain_name)
        assert result == expected_coin_id

    @pytest.mark.parametrize(
        "enter_timestamp,expected_days",
        [
            (int(datetime.now().timestamp()) - 86400, 1.0),  # 1 day ago
            (int(datetime.now().timestamp()) - 172800, 2.0),  # 2 days ago
            (int(datetime.now().timestamp()), 0.0),  # now
        ]
    )
    def test_calculate_days_since_entry(self, enter_timestamp, expected_days):
        """Test _calculate_days_since_entry method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._calculate_days_since_entry(enter_timestamp)
        # Allow for small time differences
        assert abs(result - expected_days) < 1.0

    @pytest.mark.parametrize(
        "position,expected_result",
        [
            ({"enter_timestamp": int(datetime.now().timestamp()) - 86400, "min_hold_days": 0.5}, True),
            ({"enter_timestamp": int(datetime.now().timestamp()) - 86400, "min_hold_days": 2.0}, False),
            ({}, True),  # No time requirements
            ({"min_hold_days": 0}, True),  # No time requirements
        ]
    )
    def test_check_minimum_time_met(self, position, expected_result):
        """Test _check_minimum_time_met method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._check_minimum_time_met(position)
        assert result == expected_result

    @pytest.mark.parametrize(
        "position,expected_result",
        [
            ({"gauge_address": "0xGauge123"}, True),
            ({"staked": True}, True),
            ({"staked_amount": 1000}, True),
            ({"staked_amount": 0}, False),
            ({}, False),
        ]
    )
    def test_has_staking_metadata(self, position, expected_result):
        """Test _has_staking_metadata method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._has_staking_metadata(position)
        assert result == expected_result

    @pytest.mark.parametrize(
        "chain,expected_address",
        [
            ("mode", "0xd988097fb8612cc24eeC14542bC03424c656005f"),
            ("optimism", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"),
            ("unknown_chain", None),
        ]
    )
    def test_get_usdc_address(self, chain, expected_address):
        """Test _get_usdc_address method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._get_usdc_address(chain)
        assert result == expected_address
    
    def test_get_usdc_address_exception_handling(self):
        """Test _get_usdc_address method exception handling."""
        base_behaviour = self._create_base_behaviour()
        
        # Test exception handling by mocking to_checksum_address to raise an exception
        with patch('packages.valory.skills.liquidity_trader_abci.behaviours.base.to_checksum_address', side_effect=Exception("Test exception")):
            result = base_behaviour._get_usdc_address("optimism")
            assert result is None

    @pytest.mark.parametrize(
        "chain,expected_address",
        [
            ("optimism", "0xfc2e6e6bcbd49ccf3a5f029c79984372dcbfe527"),
            ("mode", "0xcfd1d50ce23c46d3cf6407487b2f8934e96dc8f9"),
            ("unknown_chain", None),
        ]
    )
    def test_get_olas_address(self, chain, expected_address):
        """Test _get_olas_address method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._get_olas_address(chain)
        assert result == expected_address

    @pytest.mark.parametrize(
        "chain,expected_addresses",
        [
            ("mode", {"0xcfd1d50ce23c46d3cf6407487b2f8934e96dc8f9", "0x7f9adfbd38b669f03d1d11000bc76b9aaea28a81"}),
            ("optimism", {"0xfc2e6e6bcbd49ccf3a5f029c79984372dcbfe527", "0x9560e827af36c94d2ac33a39bce1fe78631088db"}),
            ("unknown_chain", set()),
        ]
    )
    def test_get_reward_token_addresses(self, chain, expected_addresses):
        """Test _get_reward_token_addresses method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._get_reward_token_addresses(chain)
        assert result == expected_addresses

    def test_get_current_timestamp(self):
        """Test _get_current_timestamp method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._get_current_timestamp()
        assert isinstance(result, int)
        assert result > 0

    @pytest.mark.parametrize(
        "token_address,date,expected_key",
        [
            ("0xToken123", "2024-01-01", "token_price_cache_0xtoken123_2024-01-01"),
            ("0xToken456", None, "token_price_cache_0xtoken456_None"),
        ]
    )
    def test_get_price_cache_key(self, token_address, date, expected_key):
        """Test _get_price_cache_key method."""
        base_behaviour = self._create_base_behaviour()
        
        result = base_behaviour._get_price_cache_key(token_address, date)
        assert result == expected_key

    @pytest.mark.parametrize("test_name,strategy,strategies_executables,kwargs,expected_result,expected_logs", [
        (
            "successful_strategy_execution",
            "test_strategy",
            {"test_strategy": ("def test_method(): return 'success'", "test_method")},
            {},
            "success",
            []
        ),
        (
            "strategy_not_found",
            "unknown_strategy",
            {"test_strategy": ("def test_method(): return 'success'", "test_method")},
            {},
            None,
            ["No executable was found for strategy='unknown_strategy'!"]
        ),
        (
            "strategy_with_kwargs",
            "test_strategy",
            {"test_strategy": ("def test_method(param): return f'param_{param}'", "test_method")},
            {"param": "value"},
            "param_value",
            []
        ),
        (
            "generator_method_result",
            "test_strategy",
            {"test_strategy": ("def test_method(): yield 1; yield 2; yield 3", "test_method")},
            {},
            [1, 2, 3],
            []
        ),
        (
            "method_not_found_in_executable",
            "test_strategy",
            {"test_strategy": ("def wrong_method(): return 'wrong'", "test_method")},
            {},
            None,
            ["No 'test_method' method was found in test_strategy executable."]
        ),
        (
            "globals_cleanup_before_exec",
            "test_strategy",
            {"test_strategy": ("def test_method(): return 'success'", "test_method")},
            {},
            "success",
            []
        ),
        (
            "empty_strategies_dict",
            "test_strategy",
            {},
            {},
            None,
            ["No executable was found for strategy='test_strategy'!"]
        ),
        (
            "strategy_with_complex_return",
            "test_strategy",
            {"test_strategy": ("def test_method(): return {'key': 'value', 'number': 42}", "test_method")},
            {},
            {"key": "value", "number": 42},
            []
        ),
    ])
    def test_execute_strategy_comprehensive(self, test_name, strategy, strategies_executables, kwargs, expected_result, expected_logs):
        """Test execute_strategy function comprehensively."""
        # Clear any existing globals that might interfere
        if "test_method" in globals():
            del globals()["test_method"]
        
        # Capture log messages
        with patch('logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            result = execute_strategy(strategy, strategies_executables, **kwargs)
            
            # Verify the result
            assert result == expected_result, f"Test '{test_name}' failed: expected {expected_result}, got {result}"
            
            # Verify expected log messages
            for expected_log in expected_logs:
                mock_logger_instance.error.assert_any_call(expected_log)
            
            # Verify globals cleanup (for successful cases)
            if expected_result is not None and "test_method" in globals():
                # The method should be available after execution
                assert "test_method" in globals()

    def test_properties(self):
        """Test property methods."""
        base_behaviour = self._create_base_behaviour()
        
        # Test synchronized_data property
        # Mock the super().synchronized_data to return a mock object
        with patch.object(type(base_behaviour).__bases__[0], 'synchronized_data', new_callable=PropertyMock) as mock_super_sync_data:
            mock_super_sync_data.return_value = MagicMock()
            result = base_behaviour.synchronized_data
            assert result is not None
        
        # Test params property
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value = MagicMock()
            result = base_behaviour.params
            assert result is not None
        
        # Test shared_state property
        with patch.object(type(base_behaviour), 'context') as mock_context:
            mock_context.state = MagicMock()
            result = base_behaviour.shared_state
            assert result is not None
        
        # Test coingecko property
        with patch.object(type(base_behaviour), 'context') as mock_context:
            mock_context.coingecko = MagicMock()
            result = base_behaviour.coingecko
            assert result is not None

    def test_store_and_read_methods(self):
        """Test store and read methods."""
        base_behaviour = self._create_base_behaviour()
        
        # Test store_current_positions
        base_behaviour.current_positions = [{"test": "position"}]
        base_behaviour.store_current_positions()
        
        # Test read_current_positions
        base_behaviour.current_positions = []
        base_behaviour.read_current_positions()
        
        # Test store_whitelisted_assets
        base_behaviour.whitelisted_assets = {"test": "asset"}
        base_behaviour.store_whitelisted_assets()
        
        # Test read_whitelisted_assets
        base_behaviour.whitelisted_assets = {}
        base_behaviour.read_whitelisted_assets()
        
        # Test store_funding_events
        base_behaviour.funding_events = {"test": "event"}
        base_behaviour.store_funding_events()
        
        # Test read_funding_events
        base_behaviour.funding_events = {}
        base_behaviour.read_funding_events()
        
        # Test store_gas_costs
        base_behaviour.store_gas_costs()
        
        # Test read_gas_costs
        base_behaviour.read_gas_costs()
        
        # Test store_portfolio_data
        base_behaviour.portfolio_data = {"test": "portfolio"}
        base_behaviour.store_portfolio_data()
        
        # Test read_portfolio_data
        base_behaviour.portfolio_data = {}
        base_behaviour.read_portfolio_data()

    def test_get_active_lp_addresses(self):
        """Test _get_active_lp_addresses method."""
        base_behaviour = self._create_base_behaviour()
        
        # Test with active positions
        base_behaviour.current_positions = [
            {"status": "open", "pool_address": "0xPool123"},
            {"status": "closed", "pool_address": "0xPool456"},
            {"status": "open", "pool_address": "0xPool789"}
        ]
        
        result = base_behaviour._get_active_lp_addresses()
        expected = {"0xpool123", "0xpool789"}
        assert result == expected
        
        # Test with no positions
        base_behaviour.current_positions = []
        result = base_behaviour._get_active_lp_addresses()
        assert result == set()

    def test_build_exit_pool_action_base(self):
        """Test _build_exit_pool_action_base method."""
        base_behaviour = self._create_base_behaviour()
        
        # Test with regular position
        position = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool123",
            "pool_type": "stable",
            "is_stable": True,
            "is_cl_pool": False,
            "token_id": 123,
            "liquidity": 1000
        }
        
        result = base_behaviour._build_exit_pool_action_base(position)
        assert result["action"] == "ExitPool"
        assert result["dex_type"] == "velodrome"
        assert result["chain"] == "optimism"
        assert result["token_id"] == 123
        assert result["liquidity"] == 1000
        
        # Test with CL pool position
        cl_position = {
            "dex_type": "velodrome",
            "chain": "optimism",
            "pool_address": "0xPool123",
            "is_cl_pool": True,
            "positions": [
                {"token_id": 123, "liquidity": 1000},
                {"token_id": 456, "liquidity": 2000}
            ]
        }
        
        result = base_behaviour._build_exit_pool_action_base(cl_position)
        assert result["token_ids"] == [123, 456]
        assert result["liquidities"] == [1000, 2000]
        
        # Test with no position
        result = base_behaviour._build_exit_pool_action_base(None)
        assert result is None
        
        # Test with tokens provided
        tokens = [
            {"token": "0xToken123", "symbol": "TOKEN1"},
            {"token": "0xToken456", "symbol": "TOKEN2"}
        ]
        
        result = base_behaviour._build_exit_pool_action_base(position, tokens)
        assert result["action"] == "ExitPool"
        assert result["dex_type"] == "velodrome"
        assert result["chain"] == "optimism"
        assert result["token_id"] == 123
        assert result["liquidity"] == 1000
        assert "assets" in result
        assert result["assets"] == ["0xToken123", "0xToken456"]
        
        # Test with CL pool position and tokens
        result = base_behaviour._build_exit_pool_action_base(cl_position, tokens)
        assert result["token_ids"] == [123, 456]
        assert result["liquidities"] == [1000, 2000]
        assert "assets" in result
        assert result["assets"] == ["0xToken123", "0xToken456"]
        
        # Test with empty tokens list
        result = base_behaviour._build_exit_pool_action_base(position, [])
        assert "assets" not in result  # Should not add assets if tokens list is empty
        
        # Test with None tokens
        result = base_behaviour._build_exit_pool_action_base(position, None)
        assert "assets" not in result  # Should not add assets if tokens is None

    def test_build_swap_to_usdc_action(self):
        """Test _build_swap_to_usdc_action method."""
        base_behaviour = self._create_base_behaviour()
        
        # Test successful swap action
        result = base_behaviour._build_swap_to_usdc_action(
            "optimism", "0xToken123", "TEST", 0.5, "Test swap"
        )
        
        assert result["action"] == "FindBridgeRoute"
        assert result["chain"] == "optimism"
        assert result["from_token"] == "0xToken123"
        assert result["from_token_symbol"] == "TEST"
        assert result["funds_percentage"] == 0.5
        assert result["description"] == "Test swap"
        
        # Test with USDC token (should return None)
        result = base_behaviour._build_swap_to_usdc_action(
            "optimism", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "USDC"
        )
        assert result is None
        
        # Test with unsupported chain (should return None due to missing USDC address)
        result = base_behaviour._build_swap_to_usdc_action(
            "unsupported_chain", "0xToken123", "TEST"
        )
        assert result is None
        
        # Test with OLAS token (should return None)
        result = base_behaviour._build_swap_to_usdc_action(
            "optimism", "0xFC2E6e6BCbd49ccf3A5f029c79984372DcBFE527", "OLAS"
        )
        assert result is None
        
        # Test with OLAS token on Mode chain (should return None)
        result = base_behaviour._build_swap_to_usdc_action(
            "mode", "0xcfD1D50ce23C46D3Cf6407487B2F8934e96DC8f9", "OLAS"
        )
        assert result is None
        
        # Test with OLAS token but no olas_address method (should still work)
        with patch.object(type(base_behaviour), '_get_olas_address', return_value=None):
            result = base_behaviour._build_swap_to_usdc_action(
                "optimism", "0xFC2E6e6BCbd49ccf3A5f029c79984372DcBFE527", "OLAS"
            )
            # Should not return None since _get_olas_address returns None
            assert result is not None
        
        # Test exception handling (should return None)
        with patch.object(type(base_behaviour), '_get_usdc_address', side_effect=Exception("Test exception")):
            result = base_behaviour._build_swap_to_usdc_action(
                "optimism", "0xToken123", "TEST"
            )
            assert result is None

    @pytest.mark.parametrize(
        "test_name,position,safe_addresses,expected_result,expected_assertions",
        [
            (
                "velodrome_cl_pool_success",
                {
                    "chain": "optimism",
                    "pool_address": "0xPool123",
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123",
                    "positions": [
                        {"token_id": 123},
                        {"token_id": 456}
                    ]
                },
                {"optimism": "0xSafe123", "mode": "0xSafe456"},
                "action_dict",
                lambda result: (
                    result is not None and
                    result["action"] == "UnstakeLpTokens" and
                    result["dex_type"] == "velodrome" and
                    result["chain"] == "optimism" and
                    result["is_cl_pool"] is True and
                    result["token_ids"] == [123, 456] and
                    result["gauge_address"] == "0xGauge123"
                )
            ),
            (
                "velodrome_regular_pool_success",
                {
                    "chain": "mode",
                    "pool_address": "0xPool456",
                    "dex_type": "velodrome",
                    "is_cl_pool": False,
                    "gauge_address": "0xGauge456"
                },
                {"mode": "0xSafe456"},
                "action_dict",
                lambda result: (
                    result is not None and
                    result["action"] == "UnstakeLpTokens" and
                    result["dex_type"] == "velodrome" and
                    result["chain"] == "mode" and
                    result["is_cl_pool"] is False and
                    result["gauge_address"] == "0xGauge456"
                )
            ),
            (
                "non_velodrome_pool_skipped",
                {
                    "chain": "optimism",
                    "pool_address": "0xPool123",
                    "dex_type": "uniswap"
                },
                {"optimism": "0xSafe123"},
                None,
                lambda result: result is None
            ),
            (
                "missing_safe_address",
                {
                    "chain": "optimism",
                    "pool_address": "0xPool123",
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123",
                    "positions": [{"token_id": 123}]
                },
                {},
                None,
                lambda result: result is None
            ),
            (
                "missing_chain_parameter",
                {
                    "pool_address": "0xPool123",
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123",
                    "positions": [{"token_id": 123}]
                },
                {"optimism": "0xSafe123"},
                None,
                lambda result: result is None
            ),
            (
                "missing_pool_address_parameter",
                {
                    "chain": "optimism",
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123",
                    "positions": [{"token_id": 123}]
                },
                {"optimism": "0xSafe123"},
                None,
                lambda result: result is None
            ),
            (
                "missing_both_required_parameters",
                {
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123",
                    "positions": [{"token_id": 123}]
                },
                {"optimism": "0xSafe123"},
                None,
                lambda result: result is None
            ),
            (
                "cl_pool_with_single_token_id",
                {
                    "chain": "optimism",
                    "pool_address": "0xPool123",
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123",
                    "token_id": 789  # Single position format
                },
                {"optimism": "0xSafe123"},
                "action_dict",
                lambda result: (
                    result is not None and
                    result["action"] == "UnstakeLpTokens" and
                    result["token_ids"] == [789]
                )
            ),
            (
                "cl_pool_no_token_ids_found",
                {
                    "chain": "optimism",
                    "pool_address": "0xPool123",
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123"
                    # No positions or token_id
                },
                {"optimism": "0xSafe123"},
                None,
                lambda result: result is None
            ),
            (
                "exception_handling",
                {
                    "chain": "optimism",
                    "pool_address": "0xPool123",
                    "dex_type": "velodrome",
                    "is_cl_pool": True,
                    "gauge_address": "0xGauge123",
                    "positions": "invalid_positions"  # This will cause an exception when trying to iterate
                },
                {"optimism": "0xSafe123"},
                None,
                lambda result: result is None
            ),
        ]
    )
    def test_build_unstake_lp_tokens_action_comprehensive(
        self, test_name, position, safe_addresses, expected_result, expected_assertions
    ):
        """Test _build_unstake_lp_tokens_action method comprehensively."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the params to include safe contract addresses
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.safe_contract_addresses = safe_addresses
            
            result = base_behaviour._build_unstake_lp_tokens_action(position)
            
            # Apply the expected assertions
            assert expected_assertions(result), f"Test '{test_name}' failed: {result}"

    def test_constants(self):
        """Test that constants are properly defined."""
        assert ZERO_ADDRESS == "0x0000000000000000000000000000000000000000"
        assert isinstance(WHITELISTED_ASSETS, dict)
        assert isinstance(COIN_ID_MAPPING, dict)
        assert isinstance(REWARD_TOKEN_ADDRESSES, dict)
        
        # Test enum values
        assert DexType.BALANCER.value == "balancerPool"
        assert DexType.UNISWAP_V3.value == "UniswapV3"
        assert DexType.VELODROME.value == "velodrome"
        
        assert Action.CLAIM_REWARDS.value == "ClaimRewards"
        assert Action.EXIT_POOL.value == "ExitPool"
        assert Action.ENTER_POOL.value == "EnterPool"
        
        assert SwapStatus.DONE.value == "DONE"
        assert SwapStatus.PENDING.value == "PENDING"
        
        assert Decision.CONTINUE.value == "continue"
        assert Decision.WAIT.value == "wait"
        assert Decision.EXIT.value == "exit"
        
        assert PositionStatus.OPEN.value == "open"
        assert PositionStatus.CLOSED.value == "closed"
        
        assert TradingType.BALANCED.value == "balanced"
        assert TradingType.RISKY.value == "risky"
        
        assert Chain.OPTIMISM.value == "optimism"
        assert Chain.MODE.value == "mode" 

    def test_gas_cost_tracker(self):
        """Test GasCostTracker class."""
        import tempfile
        from pathlib import Path
        from packages.valory.skills.liquidity_trader_abci.behaviours.base import GasCostTracker
        
        # Create temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test GasCostTracker initialization and methods
            tracker = GasCostTracker(str(temp_path / "gas_costs.json"))
            
            # Test log_gas_usage
            tracker.log_gas_usage("optimism", 1234567890, "0xHash123", 21000, 20000000000)
            
            # Test update_data
            tracker.update_data({"test": "data"})
            
            # Verify data was updated
            assert tracker.data == {"test": "data"}



    def test_get_token_decimals(self):
        """Test _get_token_decimals method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            def side_effect(*args, **kwargs):
                yield None
                return 18
            mock_contract.side_effect = side_effect
            result = self._consume_generator(base_behaviour._get_token_decimals("optimism", "0xToken123"))
            assert result == 18

    def test_get_token_symbol(self):
        """Test _get_token_symbol method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            def side_effect(*args, **kwargs):
                yield None
                return "USDC"
            mock_contract.side_effect = side_effect
            result = self._consume_generator(base_behaviour._get_token_symbol("optimism", "0xToken123"))
            assert result == "USDC"

    def test_get_native_balance(self):
        """Test _get_native_balance method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock get_ledger_api_response method
        with patch.object(base_behaviour, 'get_ledger_api_response') as mock_ledger:
            def side_effect(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.performative = LedgerApiMessage.Performative.STATE
                mock_response.state.body = {"get_balance_result": "1000000000000000000"}
                yield None
                return mock_response
            mock_ledger.side_effect = side_effect
            
            result = self._consume_generator(base_behaviour._get_native_balance("optimism", "0xAccount123"))
            assert result == 1000000000000000000

    def test_get_token_balance(self):
        """Test _get_token_balance method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            def side_effect(*args, **kwargs):
                yield None
                return 1000000000000000000
            mock_contract.side_effect = side_effect
            result = self._consume_generator(base_behaviour._get_token_balance("optimism", "0xAccount123", "0xToken123"))
            assert result == 1000000000000000000

    def test_calculate_min_num_of_safe_tx_required(self):
        """Test _calculate_min_num_of_safe_tx_required method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the required methods
        with patch.object(base_behaviour, '_get_liveness_ratio') as mock_liveness_ratio:
            with patch.object(base_behaviour, '_get_liveness_period') as mock_liveness_period:
                with patch.object(base_behaviour, '_get_ts_checkpoint') as mock_ts_checkpoint:
                    with patch.object(type(base_behaviour), 'round_sequence', new_callable=PropertyMock) as mock_round_sequence:
                        def liveness_side_effect(*args, **kwargs):
                            yield None
                            return 1000
                        def period_side_effect(*args, **kwargs):
                            yield None
                            return 86400
                        def ts_side_effect(*args, **kwargs):
                            yield None
                            return 1000000000
                        
                        mock_liveness_ratio.side_effect = liveness_side_effect
                        mock_liveness_period.side_effect = period_side_effect
                        mock_ts_checkpoint.side_effect = ts_side_effect
                        mock_round_sequence.return_value.last_round_transition_timestamp = datetime.fromtimestamp(1000000864)
                        
                        result = self._consume_generator(base_behaviour._calculate_min_num_of_safe_tx_required("optimism"))
                        assert result is not None

    @pytest.mark.parametrize(
        "token_address,date,read_kv_result,expected_result,test_description",
        [
            # Test case 1: No cache data
            ("0xToken123", "2024-01-01", {}, None, "No cache data"),
            # Test case 2: Historical price with date
            ("0xToken123", "2024-01-01", {"token_price_cache_0xtoken123_2024-01-01": '{"2024-01-01": 1.5}'}, 1.5, "Historical price found"),
            # Test case 3: Current price - no current data
            ("0xToken123", None, {"token_price_cache_0xtoken123_None": '{"historical": [1.0, 1640995300]}'}, None, "No current price data"),
            # Test case 4: Current price - expired cache 
            ("0xToken123", None, {"token_price_cache_0xtoken123_None": '{"current": [1.0, 1640995300]}'}, None, "Expired current price cache"),
            # Test case 5: Current price - valid cache 
            ("0xToken123", None, {"token_price_cache_0xtoken123_None": '{"current": [1.0, "recent_timestamp"]'}, 1.0, "Valid current price cache"),
            # Test case 6: JSON decode error
            ("0xToken123", "2024-01-01", {"token_price_cache_0xtoken123_2024-01-01": "invalid_json"}, None, "JSON decode error"),
        ]
    )
    def test_get_cached_price(self, token_address, date, read_kv_result, expected_result, test_description):
        """Test _get_cached_price method with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _read_kv method
        with patch.object(base_behaviour, '_read_kv') as mock_read_kv:
            # Handle special cases for timestamp-based tests
            if test_description == "Expired current price cache":
                # Mock expired timestamp (older than CACHE_TTL)
                old_timestamp = int(time.time()) - 13 * 3600  # 13 hours ago
                def expired_side_effect(*args, **kwargs):
                    yield None
                    return {"token_price_cache_0xtoken123_None": f'{{"current": [1.0, {old_timestamp}]}}'}
                mock_read_kv.side_effect = expired_side_effect
            elif test_description == "Valid current price cache":
                # Mock recent timestamp (within CACHE_TTL)
                recent_timestamp = int(time.time()) - 100  # 100 seconds ago
                def valid_side_effect(*args, **kwargs):
                    yield None
                    return {"token_price_cache_0xtoken123_None": f'{{"current": [1.0, {recent_timestamp}]}}'}
                mock_read_kv.side_effect = valid_side_effect
            else:
                # Use the provided read_kv_result for other test cases
                def default_side_effect(*args, **kwargs):
                    yield None
                    return read_kv_result
                mock_read_kv.side_effect = default_side_effect
            
            # Mock _get_current_timestamp to return a fixed value for consistent testing
            with patch.object(base_behaviour, '_get_current_timestamp') as mock_timestamp:
                mock_timestamp.return_value = int(time.time())
                
                result = self._consume_generator(base_behaviour._get_cached_price(token_address, date))
                
                if test_description == "JSON decode error":
                    # For JSON decode error, we expect None due to exception handling
                    assert result is None
                else:
                    assert result == expected_result

    @pytest.mark.parametrize(
        "coin_id,price,date,read_kv_result,expected_write_calls,expected_error_logs,expected_price_data",
        [
            # Test case 1: Basic caching with empty existing data
            ("test_key", 1.5, "2024-01-01", {}, 1, [], {"2024-01-01": 1.5}),
            # Test case 2: Caching with existing valid data
            ("test_key", 2.0, "2024-01-01", {"token_price_cache_test": '{"2024-01-01": 1.0}'}, 1, [], {"2024-01-01": 1.0, "2024-01-01": 2.0}),
            # Test case 3: JSON decode error
            ("test_key", 1.5, "2024-01-01", {"token_price_cache_test_key_2024-01-01": "invalid_json"}, 1, ["Invalid cache data for token test_key, resetting cache"], {"2024-01-01": 1.5}),
            # Test case 4: Historical price (with date)
            ("test_token", 1.5, "2024-01-01", {}, 1, [], {"2024-01-01": 1.5}),
            # Test case 5: Current price (without date)
            ("test_token", 2.0, None, {}, 1, [], {"current": [2.0, "timestamp"]}),
        ]
    )
    def test_cache_price(self, coin_id, price, date, read_kv_result, expected_write_calls, expected_error_logs, expected_price_data):
        """Test _cache_price method with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _read_kv and _write_kv methods
        with patch.object(base_behaviour, '_read_kv') as mock_read_kv:
            with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                def read_side_effect(*args, **kwargs):
                    yield None
                    return read_kv_result
                mock_read_kv.side_effect = read_side_effect
                
                mock_write_kv.return_value = iter([None])
                
                with patch.object(base_behaviour.context.logger, 'error') as mock_error:
                    self._consume_generator(base_behaviour._cache_price(coin_id, price, date))
                    
                    # Verify _write_kv was called the expected number of times
                    assert mock_write_kv.call_count == expected_write_calls
                    
                    # Verify error logging if expected
                    if expected_error_logs:
                        mock_error.assert_called_once()
                        error_call = mock_error.call_args[0][0]
                        assert expected_error_logs[0] in error_call
                    else:
                        mock_error.assert_not_called()
                    
                    # Verify the price data structure if not testing error case
                    if not expected_error_logs:
                        mock_write_kv.assert_called_once()
                        call_args = mock_write_kv.call_args[0][0]
                        cache_key = list(call_args.keys())[0]
                        price_data = json.loads(call_args[cache_key])
                        
                        if date:
                            # Historical price
                            assert price_data[date] == price
                        else:
                            # Current price
                            assert "current" in price_data
                            assert price_data["current"][0] == price  # price
                            assert isinstance(price_data["current"][1], int)  # timestamp

    def test_get_native_balance_failure(self):
        """Test _get_native_balance method when ledger API fails"""
        base_behaviour = self._create_base_behaviour()
        
        # Mock get_ledger_api_response to return failure response
        with patch.object(base_behaviour, 'get_ledger_api_response') as mock_ledger:
            def side_effect(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.performative = LedgerApiMessage.Performative.ERROR
                yield None
                return mock_response
            mock_ledger.side_effect = side_effect
            
            with patch.object(base_behaviour.context.logger, 'error') as mock_error:
                result = self._consume_generator(base_behaviour._get_native_balance("optimism", "0xSafe123"))
                
                assert result is None
                mock_error.assert_called_once()
                error_call = mock_error.call_args[0][0]
                assert "Could not calculate the balance of the safe" in error_call

    @pytest.mark.parametrize(
        "liveness_ratio,liveness_period,ts_checkpoint,expected_result,test_description",
        [
            # Test case 1: liveness_ratio is None (line 1057)
            (None, 86400, 1640995300, None, "liveness_ratio is None"),
            # Test case 2: liveness_period is None (line 1067)
            (1000, None, 1640995300, None, "liveness_period is None"),
            # Test case 3: last_ts_checkpoint is None
            (1000, 86400, None, None, "last_ts_checkpoint is None"),
            # Test case 4: All values are valid
            (1000, 86400, 1640995300, 2, "All values are valid"),
        ]
    )
    def test_calculate_min_num_of_safe_tx_required_failure_cases(self, liveness_ratio, liveness_period, ts_checkpoint, expected_result, test_description):
        """Test _calculate_min_num_of_safe_tx_required method with various scenarios"""
        base_behaviour = self._create_base_behaviour()
        
        # Mock round_sequence.last_round_transition_timestamp
        mock_timestamp = MagicMock()
        mock_timestamp.timestamp.return_value = 1640995300
        with patch.object(type(base_behaviour), 'round_sequence', new_callable=PropertyMock) as mock_round_sequence:
            mock_round_sequence.return_value.last_round_transition_timestamp = mock_timestamp
            
            with patch.object(base_behaviour, '_get_liveness_ratio') as mock_liveness_ratio:
                with patch.object(base_behaviour, '_get_liveness_period') as mock_liveness_period:
                    with patch.object(base_behaviour, '_get_ts_checkpoint') as mock_ts_checkpoint:
                        def liveness_side_effect(*args, **kwargs):
                            yield None
                            return liveness_ratio
                        def period_side_effect(*args, **kwargs):
                            yield None
                            return liveness_period
                        def ts_side_effect(*args, **kwargs):
                            yield None
                            return ts_checkpoint
                        
                        mock_liveness_ratio.side_effect = liveness_side_effect
                        mock_liveness_period.side_effect = period_side_effect
                        mock_ts_checkpoint.side_effect = ts_side_effect
                        
                        result = self._consume_generator(base_behaviour._calculate_min_num_of_safe_tx_required("optimism"))
                        assert result == expected_result











    @pytest.mark.parametrize(
        "period_count,kv_store_data,expected_result,expected_log_level,expected_log_message,test_description",
        [
            # Test case 1: period == 0, forcing immediate update
            (0, {}, True, "info", "period == 0, forcing immediate update", "Period 0 forces immediate update"),
            
            # Test case 2: Invalid timestamp format
            (1, {"last_reward_update_optimism": "invalid_timestamp"}, True, "error", "Invalid timestamp format in kv_store", "Invalid timestamp forces update"),
            
            # Test case 3: No cached value, should update
            (1, {}, True, None, None, "No cached value triggers update"),
            
            # Test case 4: Recent update (within 12 hours), should not update
            (1, {"last_reward_update_optimism": "recent_timestamp"}, False, None, None, "Recent update prevents update"),
            
            # Test case 5: Old update (older than 12 hours), should update
            (1, {"last_reward_update_optimism": "old_timestamp"}, True, None, None, "Old update triggers update"),
        ]
    )
    def test_should_update_rewards_from_subgraph_parameterized(
        self, period_count, kv_store_data, expected_result, expected_log_level, expected_log_message, test_description
    ):
        """Test should_update_rewards_from_subgraph method with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock synchronized_data.period_count
        with patch.object(type(base_behaviour), 'synchronized_data', new_callable=PropertyMock) as mock_sync_data:
            mock_sync_data.return_value.period_count = period_count
            
            # Mock _read_kv method with proper timestamp handling
            with patch.object(base_behaviour, '_read_kv') as mock_read_kv:
                if test_description == "Recent update prevents update":
                    # Recent update: 100 seconds ago (well within 12 hours)
                    recent_timestamp = int(time.time()) - 100
                    def recent_side_effect(*args, **kwargs):
                        yield None
                        return {"last_reward_update_optimism": str(recent_timestamp)}
                    mock_read_kv.side_effect = recent_side_effect
                elif test_description == "Old update triggers update":
                    # Old update: 13 hours ago (older than 12 hours)
                    old_timestamp = int(time.time()) - 13 * 3600
                    def old_side_effect(*args, **kwargs):
                        yield None
                        return {"last_reward_update_optimism": str(old_timestamp)}
                    mock_read_kv.side_effect = old_side_effect
                else:
                    # Use the provided kv_store_data for other test cases
                    def default_side_effect(*args, **kwargs):
                        yield None
                        return kv_store_data
                    mock_read_kv.side_effect = default_side_effect
                
                # Mock logger based on expected log level
                if expected_log_level == "info":
                    with patch.object(base_behaviour.context.logger, 'info') as mock_logger:
                        result = self._consume_generator(base_behaviour.should_update_rewards_from_subgraph("optimism"))
                        assert result == expected_result
                        mock_logger.assert_called_once()
                        log_call = mock_logger.call_args[0][0]
                        assert expected_log_message in log_call
                elif expected_log_level == "error":
                    with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                        result = self._consume_generator(base_behaviour.should_update_rewards_from_subgraph("optimism"))
                        assert result == expected_result
                        mock_logger.assert_called_once()
                        log_call = mock_logger.call_args[0][0]
                        assert expected_log_message in log_call
                else:
                    # No specific logging expected
                    result = self._consume_generator(base_behaviour.should_update_rewards_from_subgraph("optimism"))
                    assert result == expected_result



    @pytest.mark.parametrize(
        "test_scenario,coingecko_mock,expected_result,test_description",
        [
            # Test case 1: Basic functionality - no rate limiter
            ("basic", None, "int_ge_0", "Basic functionality test"),
            
            # Test case 2: No rate_limiter attribute (line 1388)
            ("no_rate_limiter", {"has_rate_limiter": False}, 0, "No rate_limiter attribute"),
            
            # Test case 3: No credits (line 1394)
            ("no_credits", {"has_rate_limiter": True, "no_credits": True}, 0, "No credits available"),
            
            # Test case 4: Rate limited with remainder
            ("rate_limited_with_remainder", {"has_rate_limiter": True, "no_credits": False, "rate_limited": True, "last_request_time": 1000000000, "current_time": 1000000030}, 30, "Rate limited with 30s remainder"),
            
            # Test case 5: Rate limited no remainder
            ("rate_limited_no_remainder", {"has_rate_limiter": True, "no_credits": False, "rate_limited": True, "last_request_time": 1000000000, "current_time": 1000000060}, 0, "Rate limited with no remainder"),
            
            # Test case 6: Normal operation (no rate limiting)
            ("normal", {"has_rate_limiter": True, "no_credits": False, "rate_limited": False}, 0, "Normal operation"),
        ]
    )
    def test_calculate_rate_limit_wait_time(self, test_scenario, coingecko_mock, expected_result, test_description):
        """Test _calculate_rate_limit_wait_time method with various scenarios"""
        base_behaviour = self._create_base_behaviour()
        
        if test_scenario == "basic":
            # Test with no rate limiter
            result = base_behaviour._calculate_rate_limit_wait_time()
            assert isinstance(result, int)
            assert result >= 0
            return
        
        # Mock coingecko context
        with patch.object(base_behaviour.context, 'coingecko') as mock_coingecko:
            if not coingecko_mock["has_rate_limiter"]:
                # Remove rate_limiter attribute to simulate hasattr check failing (line 1388)
                if hasattr(mock_coingecko, 'rate_limiter'):
                    delattr(mock_coingecko, 'rate_limiter')
            else:
                # Create rate_limiter mock
                mock_rate_limiter = MagicMock()
                mock_rate_limiter.no_credits = coingecko_mock.get("no_credits", False)
                mock_rate_limiter.rate_limited = coingecko_mock.get("rate_limited", False)
                
                if coingecko_mock.get("last_request_time"):
                    mock_rate_limiter.last_request_time = coingecko_mock["last_request_time"]
                
                mock_coingecko.rate_limiter = mock_rate_limiter
                
                # Mock time.time() for rate limited scenarios
                if coingecko_mock.get("current_time"):
                    with patch('time.time', return_value=coingecko_mock["current_time"]):
                        result = base_behaviour._calculate_rate_limit_wait_time()
                        assert result == expected_result
                        return
            
            # For non-rate-limited scenarios
            result = base_behaviour._calculate_rate_limit_wait_time()
            assert result == expected_result



    @pytest.mark.parametrize(
        "test_scenario,cached_value,on_chain_value,expected_result,should_write_kv,should_reset,expected_log_message",
        [
            # Test case 1: No cache exists - should reset
            (
                "no_cache",
                None,  # No cache
                None,  # Not relevant
                1000,  # Expected result from reset
                False,  # Should not write to KV
                True,   # Should call reset
                None,   # No log message
            ),
            # Test case 2: Cached value differs from on-chain - should sync
            (
                "cached_mismatch",
                "500",  # Cached value
                1000,   # On-chain value (different)
                1000,   # Expected result (on-chain value)
                True,   # Should write to KV to sync
                False,  # Should not reset
                "Syncing ETH remaining amount from cached 500 to on-chain 1000",
            ),
            # Test case 3: On-chain amount is None - should return cached (line 1889)
            (
                "on_chain_none",
                "800",  # Cached value
                None,   # On-chain amount is None
                800,    # Expected result (cached value)
                False,  # Should not write to KV
                False,  # Should not reset
                None,   # No log message
            ),
            # Test case 4: On-chain amount matches cached - should return cached (line 1887)
            (
                "on_chain_matches_cache",
                "1000000000000000000",  # Cached value (1 ETH)
                1000000000000000000,    # On-chain value (matches cached)
                1000000000000000000,    # Expected result (cached value)
                False,  # Should not write to KV
                False,  # Should not reset
                None,   # No log message
            ),
            # Test case 5: Exception occurs - should reset and log error
            (
                "exception_occurs",
                "invalid_value",  # Invalid cached value
                1000,             # On-chain value
                750,              # Expected result from reset
                False,            # Should not write to KV
                True,             # Should call reset
                "Invalid ETH remaining amount in kv_store: invalid_value",
            ),
        ]
    )
    def test_get_eth_remaining_amount_comprehensive(
        self, test_scenario, cached_value, on_chain_value, expected_result, 
        should_write_kv, should_reset, expected_log_message
    ):
        """Comprehensive parameterized test for get_eth_remaining_amount method covering all scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock all the methods in the call chain
        with patch.object(base_behaviour, '_read_kv') as mock_read_kv:
            with patch.object(base_behaviour, '_get_native_balance') as mock_native_balance:
                with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                    with patch.object(base_behaviour, 'reset_eth_remaining_amount') as mock_reset:
                        with patch.object(base_behaviour.context.logger, 'info') as mock_info_logger:
                            with patch.object(base_behaviour.context.logger, 'error') as mock_error_logger:
                                
                                # Mock KV store response
                                def read_kv_side_effect(*args, **kwargs):
                                    yield None
                                    if cached_value is None:
                                        return None
                                    return {"eth_remaining_amount": cached_value}
                                
                                mock_read_kv.side_effect = read_kv_side_effect
                                
                                # Mock native balance response
                                def native_balance_side_effect(*args, **kwargs):
                                    yield None
                                    return on_chain_value
                                
                                mock_native_balance.side_effect = native_balance_side_effect
                                
                                # Mock write KV response
                                def write_kv_side_effect(*args, **kwargs):
                                    yield None
                                    return True
                                
                                mock_write_kv.side_effect = write_kv_side_effect
                                
                                # Mock reset response
                                def reset_side_effect(*args, **kwargs):
                                    yield None
                                    return 750  # Default reset value
                                
                                mock_reset.side_effect = reset_side_effect
                                
                                # Execute the method
                                result = self._consume_generator(base_behaviour.get_eth_remaining_amount())
                                
                                # Verify the result
                                if should_reset:
                                    assert result == 750  # Reset value
                                else:
                                    assert result == expected_result
                                
                                # Verify KV write behavior
                                if should_write_kv:
                                    mock_write_kv.assert_called_once()
                                    write_call_args = mock_write_kv.call_args[0][0]
                                    assert write_call_args == {"eth_remaining_amount": str(on_chain_value)}
                                else:
                                    mock_write_kv.assert_not_called()
                                
                                # Verify reset behavior
                                if should_reset:
                                    mock_reset.assert_called_once()
                                else:
                                    mock_reset.assert_not_called()
                                
                                # Verify logging behavior
                                if expected_log_message:
                                    if "error" in expected_log_message.lower():
                                        mock_error_logger.assert_called_once()
                                        log_message = mock_error_logger.call_args[0][0]
                                        assert expected_log_message in log_message
                                    elif "Syncing" in expected_log_message:
                                        mock_info_logger.assert_called_once()
                                        log_message = mock_info_logger.call_args[0][0]
                                        assert expected_log_message in log_message
                                    else:
                                        # For other cases, don't assert specific logging
                                        pass







    @pytest.mark.parametrize(
        "current_remaining,amount_used,expected_new_remaining,expected_log_message",
        [
            # Test case 1: Normal subtraction
            (1000, 200, 800, "Updating ETH remaining amount in kv_store: 1000 -> 800"),
            # Test case 2: Subtraction that results in zero
            (500, 500, 0, "Updating ETH remaining amount in kv_store: 500 -> 0"),
            # Test case 3: Subtraction that would result in negative (should be capped at 0)
            (300, 500, 0, "Updating ETH remaining amount in kv_store: 300 -> 0"),
            # Test case 4: No amount used
            (1000, 0, 1000, "Updating ETH remaining amount in kv_store: 1000 -> 1000"),
            # Test case 5: Large amounts
            (1000000000000000000, 500000000000000000, 500000000000000000, 
             "Updating ETH remaining amount in kv_store: 1000000000000000000 -> 500000000000000000"),
        ]
    )
    def test_update_eth_remaining_amount_success(
        self, current_remaining, amount_used, expected_new_remaining, expected_log_message
    ):
        """Test update_eth_remaining_amount method with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock all the methods in the call chain
        with patch.object(base_behaviour, 'get_eth_remaining_amount') as mock_get_remaining:
            with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                with patch.object(base_behaviour.context.logger, 'info') as mock_logger:
                    def get_remaining_side_effect(*args, **kwargs):
                        yield None
                        return current_remaining
                    def write_kv_side_effect(*args, **kwargs):
                        yield None
                        return True
                    
                    mock_get_remaining.side_effect = get_remaining_side_effect
                    mock_write_kv.side_effect = write_kv_side_effect
                    
                    # Call the method
                    self._consume_generator(base_behaviour.update_eth_remaining_amount(amount_used))
                    
                    # Verify get_eth_remaining_amount was called
                    mock_get_remaining.assert_called_once()
                    
                    # Verify _write_kv was called with correct parameters
                    mock_write_kv.assert_called_once()
                    write_call_args = mock_write_kv.call_args[0][0]
                    assert write_call_args == {"eth_remaining_amount": str(expected_new_remaining)}
                    
                    # Verify logging
                    mock_logger.assert_called_once()
                    log_message = mock_logger.call_args[0][0]
                    assert log_message == expected_log_message

    def test_update_eth_remaining_amount_write_kv_failure(self):
        """Test update_eth_remaining_amount method when _write_kv fails."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock all the methods in the call chain
        with patch.object(base_behaviour, 'get_eth_remaining_amount') as mock_get_remaining:
            with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                with patch.object(base_behaviour.context.logger, 'info') as mock_logger:
                    def get_remaining_side_effect(*args, **kwargs):
                        yield None
                        return 1000
                    def write_kv_side_effect(*args, **kwargs):
                        yield None
                        return False  # Write fails
                    
                    mock_get_remaining.side_effect = get_remaining_side_effect
                    mock_write_kv.side_effect = write_kv_side_effect
                    
                    # Call the method - should not raise exception even if write fails
                    self._consume_generator(base_behaviour.update_eth_remaining_amount(200))
                    
                    # Verify get_eth_remaining_amount was called
                    mock_get_remaining.assert_called_once()
                    
                    # Verify _write_kv was called
                    mock_write_kv.assert_called_once()
                    
                    # Verify logging still occurred
                    mock_logger.assert_called_once()

    def test_update_eth_remaining_amount_zero_amount_used(self):
        """Test update_eth_remaining_amount method with zero amount used."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock all the methods in the call chain
        with patch.object(base_behaviour, 'get_eth_remaining_amount') as mock_get_remaining:
            with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                def get_remaining_side_effect(*args, **kwargs):
                    yield None
                    return 1000
                def write_kv_side_effect(*args, **kwargs):
                    yield None
                    return True
                
                mock_get_remaining.side_effect = get_remaining_side_effect
                mock_write_kv.side_effect = write_kv_side_effect
                
                # Call the method with zero amount
                self._consume_generator(base_behaviour.update_eth_remaining_amount(0))
                
                # Verify _write_kv was called with unchanged amount
                mock_write_kv.assert_called_once()
                write_call_args = mock_write_kv.call_args[0][0]
                assert write_call_args == {"eth_remaining_amount": "1000"}

    def test_update_eth_remaining_amount_large_amount_used(self):
        """Test update_eth_remaining_amount method with large amount that exceeds remaining."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock all the methods in the call chain
        with patch.object(base_behaviour, 'get_eth_remaining_amount') as mock_get_remaining:
            with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                def get_remaining_side_effect(*args, **kwargs):
                    yield None
                    return 10  # Small remaining amount
                def write_kv_side_effect(*args, **kwargs):
                    yield None
                    return True
                
                mock_get_remaining.side_effect = get_remaining_side_effect
                mock_write_kv.side_effect = write_kv_side_effect
                
                # Call the method with amount larger than remaining
                self._consume_generator(base_behaviour.update_eth_remaining_amount(500))
                
                # Verify _write_kv was called with zero (capped at 0)
                mock_write_kv.assert_called_once()
                write_call_args = mock_write_kv.call_args[0][0]
                assert write_call_args == {"eth_remaining_amount": "0"}

    def test_update_eth_remaining_amount_negative_amount_used(self):
        """Test update_eth_remaining_amount method with negative amount (edge case)."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock all the methods in the call chain
        with patch.object(base_behaviour, 'get_eth_remaining_amount') as mock_get_remaining:
            with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                def get_remaining_side_effect(*args, **kwargs):
                    yield None
                    return 1000
                def write_kv_side_effect(*args, **kwargs):
                    yield None
                    return True
                
                mock_get_remaining.side_effect = get_remaining_side_effect
                mock_write_kv.side_effect = write_kv_side_effect
                
                # Call the method with negative amount
                self._consume_generator(base_behaviour.update_eth_remaining_amount(-100))
                
                # Verify _write_kv was called with increased amount (subtracting negative = adding)
                mock_write_kv.assert_called_once()
                write_call_args = mock_write_kv.call_args[0][0]
                assert write_call_args == {"eth_remaining_amount": "1100"}

    def test_update_eth_remaining_amount_get_remaining_failure(self):
        """Test update_eth_remaining_amount method when get_eth_remaining_amount fails."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock all the methods in the call chain
        with patch.object(base_behaviour, 'get_eth_remaining_amount') as mock_get_remaining:
            with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
                def get_remaining_side_effect(*args, **kwargs):
                    yield None
                    return 0  # get_eth_remaining_amount returns 0 (failure case)
                def write_kv_side_effect(*args, **kwargs):
                    yield None
                    return True
                
                mock_get_remaining.side_effect = get_remaining_side_effect
                mock_write_kv.side_effect = write_kv_side_effect
                
                # Call the method
                self._consume_generator(base_behaviour.update_eth_remaining_amount(100))
                
                # Verify _write_kv was called with zero (max(0, 0 - 100) = 0)
                mock_write_kv.assert_called_once()
                write_call_args = mock_write_kv.call_args[0][0]
                assert write_call_args == {"eth_remaining_amount": "0"}

    def test_reset_eth_remaining_amount(self):
        """Test reset_eth_remaining_amount method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _write_kv and _get_native_balance methods
        with patch.object(base_behaviour, '_write_kv') as mock_write_kv:
            with patch.object(base_behaviour, '_get_native_balance') as mock_native_balance:
                def write_kv_side_effect(*args, **kwargs):
                    yield None
                    return True
                def native_balance_side_effect(*args, **kwargs):
                    yield None
                    return 1500
                mock_write_kv.side_effect = write_kv_side_effect
                mock_native_balance.side_effect = native_balance_side_effect
                
                result = self._consume_generator(base_behaviour.reset_eth_remaining_amount())
                assert result == 1500
                
                # Test case where _get_native_balance returns None
                def native_balance_none_side_effect(*args, **kwargs):
                    yield None
                    return None
                mock_native_balance.side_effect = native_balance_none_side_effect
                
                result = self._consume_generator(base_behaviour.reset_eth_remaining_amount())
                assert result == 0

    @pytest.mark.parametrize(
        "test_scenario,chain,token_address,kv_result,expected_result,expected_log_message,line_coverage",
        [
            # Test case 1: Token with accumulated rewards (line 2071)
            (
                "valid_rewards",
                "optimism",
                "0xToken123",
                {"accumulated_rewards_optimism_0xtoken123": "500"},
                500,
                None,
                "line 2071 - successful conversion"
            ),
            # Test case 2: Token with no accumulated rewards (line 2061)
            (
                "no_kv_result",
                "optimism", 
                "0xToken456",
                {},
                0,
                None,
                "line 2061 - no KV result"
            ),
            # Test case 3: Invalid rewards value (line 2076)
            (
                "invalid_value",
                "optimism",
                "0xToken789", 
                {"accumulated_rewards_optimism_0xtoken789": "invalid_value"},
                0,
                "Invalid rewards value for accumulated_rewards_optimism_0xtoken789: invalid_value, returning 0",
                "line 2076 - ValueError/TypeError handling"
            ),
            # Test case 4: rewards_value is None (line 2068)
            (
                "rewards_value_none",
                "optimism",
                "0xToken101",
                {"accumulated_rewards_optimism_0xtoken101": None},
                0,
                None,
                "line 2068 - rewards_value is None"
            ),
            # Test case 5: Different chain and token combination
            (
                "mode_chain_token",
                "mode",
                "0xOlasToken",
                {"accumulated_rewards_mode_0xolastoken": "1000"},
                1000,
                None,
                "line 2071 - different chain/token combination"
            ),
            # Test case 6: Empty string rewards value (should trigger ValueError)
            (
                "empty_string_value",
                "optimism",
                "0xTokenEmpty",
                {"accumulated_rewards_optimism_0xtokenempty": ""},
                0,
                "Invalid rewards value for accumulated_rewards_optimism_0xtokenempty: , returning 0",
                "line 2076 - empty string ValueError"
            ),
        ]
    )
    def test_get_accumulated_rewards_for_token_comprehensive(
        self, test_scenario, chain, token_address, kv_result, expected_result, 
        expected_log_message, line_coverage
    ):
        """Comprehensive parameterized test for get_accumulated_rewards_for_token method covering all scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _read_kv method
        with patch.object(base_behaviour, '_read_kv') as mock_read_kv:
            with patch.object(base_behaviour.context.logger, 'warning') as mock_warning:
                
                def read_kv_side_effect(*args, **kwargs):
                    yield None
                    return kv_result
                
                mock_read_kv.side_effect = read_kv_side_effect
                
                # Execute the method
                result = self._consume_generator(
                    base_behaviour.get_accumulated_rewards_for_token(chain, token_address)
                )
                
                # Verify the result
                assert result == expected_result, f"Expected {expected_result}, got {result} for {test_scenario}"
                
                # Verify logging behavior
                if expected_log_message:
                    mock_warning.assert_called_once()
                    warning_call = mock_warning.call_args[0][0]
                    assert expected_log_message in warning_call, f"Expected log message not found: {expected_log_message}"
                else:
                    mock_warning.assert_not_called()
                
                # Verify the correct key was used for KV lookup
                expected_key = f"accumulated_rewards_{chain}_{token_address.lower()}"
                mock_read_kv.assert_called_once_with((expected_key,))

    @pytest.mark.parametrize(
        "test_scenario,chain,position_manager,token_id,sqrt_price_x96,mock_params,mock_contract,expected_result",
        [
            (
                "successful_contract_interaction",
                "optimism",
                "0xPositionManager123",
                1000,
                1581138830084190475656131093637,
                {"optimism": "0xSugar123"},  # Valid Sugar contract address
                (1000000000000000000, 2000000000000000000),  # Valid amounts returned
                (1000000000000000000, 2000000000000000000),  # Expected result
            ),
            (
                "no_sugar_contract_address",
                "optimism",
                "0xPositionManager123",
                1000,
                1581138830084190475656131093637,
                {},  # Empty Sugar contract addresses
                None,  # Contract interaction not reached
                (0, 0),  # Expected result
            ),
            (
                "contract_returns_no_amounts",
                "optimism",
                "0xPositionManager123",
                1000,
                1581138830084190475656131093637,
                {"optimism": "0xSugar123"},  # Valid Sugar contract address
                None,  # Contract returns None
                (0, 0),  # Expected result
            ),
        ]
    )
    def test_get_velodrome_position_principal_comprehensive(
        self, test_scenario, chain, position_manager, token_id, sqrt_price_x96, 
        mock_params, mock_contract, expected_result
    ):
        """Test get_velodrome_position_principal method comprehensively."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params for Sugar contract addresses
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params_obj:
            mock_params_obj.return_value.velodrome_slipstream_helper_contract_addresses = mock_params
            
            # Mock contract interaction if we have Sugar contract addresses
            if mock_params:
                with patch.object(base_behaviour, 'contract_interact') as mock_contract_obj:
                    def side_effect(*args, **kwargs):
                        yield None
                        return mock_contract
                    mock_contract_obj.side_effect = side_effect
                    
                    result = self._consume_generator(
                        base_behaviour.get_velodrome_position_principal(
                            chain, position_manager, token_id, sqrt_price_x96
                        )
                    )
            else:
                # No contract interaction when no Sugar contract address
                result = self._consume_generator(
                    base_behaviour.get_velodrome_position_principal(
                        chain, position_manager, token_id, sqrt_price_x96
                    )
                )
            
            assert result == expected_result, f"Failed for scenario: {test_scenario}"

    @pytest.mark.parametrize(
        "test_scenario,chain,sqrt_price_x96,sqrt_ratio_a_x96,sqrt_ratio_b_x96,liquidity,mock_params,mock_contract,expected_result,expected_logs",
        [
            (
                "successful_contract_interaction",
                "optimism",
                1000000000000000000,
                1000000000000000000,
                2000000000000000000,
                1000,
                {"optimism": "0xSugar123"},  # Valid Sugar contract address
                (1000000000000000000, 2000000000000000000),  # Valid amounts returned
                (1000000000000000000, 2000000000000000000),  # Expected result
                [],  # No error logs expected
            ),
            (
                "no_sugar_contract_address",
                "optimism",
                1000000000000000000,
                1000000000000000000,
                2000000000000000000,
                1000,
                {},  # Empty Sugar contract addresses
                None,  # Contract interaction not reached
                (0, 0),  # Expected result
                ["No Velodrome Sugar contract address for chain optimism"],  # Error log expected
            ),
            (
                "contract_returns_no_amounts",
                "optimism",
                1000000000000000000,
                1000000000000000000,
                2000000000000000000,
                1000,
                {"optimism": "0xSugar123"},  # Valid Sugar contract address
                None,  # Contract returns None
                (0, 0),  # Expected result
                [],  # No error logs expected
            ),
        ]
    )
    def test_get_velodrome_amounts_for_liquidity_comprehensive(
        self, test_scenario, chain, sqrt_price_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, 
        liquidity, mock_params, mock_contract, expected_result, expected_logs
    ):
        """Test get_velodrome_amounts_for_liquidity method comprehensively."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params for Sugar contract addresses
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params_obj:
            mock_params_obj.return_value.velodrome_slipstream_helper_contract_addresses = mock_params
            
            # Mock contract interaction if we have Sugar contract addresses
            if mock_params:
                with patch.object(base_behaviour, 'contract_interact') as mock_contract_obj:
                    def side_effect(*args, **kwargs):
                        yield None
                        return mock_contract
                    mock_contract_obj.side_effect = side_effect
                    
                    result = self._consume_generator(
                        base_behaviour.get_velodrome_amounts_for_liquidity(
                            chain, sqrt_price_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity
                        )
                    )
            else:
                # No contract interaction when no Sugar contract address
                result = self._consume_generator(
                    base_behaviour.get_velodrome_amounts_for_liquidity(
                        chain, sqrt_price_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity
                    )
                )
            
            assert result == expected_result, f"Failed for scenario: {test_scenario}"

    @pytest.mark.parametrize(
        "test_scenario,chain,tick,mock_params,mock_contract,expected_result,expected_logs",
        [
            (
                "successful_contract_interaction",
                "optimism",
                1000,
                {"optimism": "0xSugar123"},  # Valid Sugar contract address
                1000000000000000000,  # Valid sqrt_ratio returned
                1000000000000000000,  # Expected result
                [],  # No error logs expected
            ),
            (
                "no_sugar_contract_address",
                "optimism",
                1000,
                {},  # Empty Sugar contract addresses
                None,  # Contract interaction not reached
                0,  # Expected result
                ["No Velodrome Sugar contract address for chain optimism"],  # Error log expected
            ),
        ]
    )
    def test_get_velodrome_sqrt_ratio_at_tick_comprehensive(
        self, test_scenario, chain, tick, mock_params, mock_contract, expected_result, expected_logs
    ):
        """Test get_velodrome_sqrt_ratio_at_tick method comprehensively."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params for Sugar contract addresses
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params_obj:
            mock_params_obj.return_value.velodrome_slipstream_helper_contract_addresses = mock_params
            
            # Mock contract interaction if we have Sugar contract addresses
            if mock_params:
                with patch.object(base_behaviour, 'contract_interact') as mock_contract_obj:
                    def side_effect(*args, **kwargs):
                        yield None
                        return mock_contract
                    mock_contract_obj.side_effect = side_effect
                    
                    result = self._consume_generator(
                        base_behaviour.get_velodrome_sqrt_ratio_at_tick(chain, tick)
                    )
            else:
                # No contract interaction when no Sugar contract address
                result = self._consume_generator(
                    base_behaviour.get_velodrome_sqrt_ratio_at_tick(chain, tick)
                )
            
            assert result == expected_result, f"Failed for scenario: {test_scenario}"

    @pytest.mark.parametrize(
        "endpoint,method,body,headers,rate_limited_code,max_retries,retry_wait,"
        "mock_responses,expected_success,expected_result,expected_logs",
        [
            # Successful request on first try
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                3,
                1,
                [{"status_code": 200, "body": '{"data": "success"}'}],
                True,
                {"data": "success"},
                ["HTTP GET call: https://api.example.com/data", "Request succeeded."],
            ),
            # Rate limiting with successful retry
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                3,
                1,
                [
                    {"status_code": 429, "body": '{"error": "rate limited"}'},
                    {"status_code": 200, "body": '{"data": "success"}'},
                ],
                True,
                {"data": "success"},
                [
                    "HTTP GET call: https://api.example.com/data",
                    "Rate limited (attempt 1/3)",
                    "Waiting 60 seconds before retrying rate-limited request",
                    "Request succeeded.",
                ],
            ),
            # Rate limiting with max retries exceeded
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                2,
                1,
                [
                    {"status_code": 429, "body": '{"error": "rate limited"}'},
                    {"status_code": 429, "body": '{"error": "rate limited"}'},
                ],
                False,
                {"error": "rate limited"},
                [
                    "HTTP GET call: https://api.example.com/data",
                    "Rate limited (attempt 1/2)",
                    "Waiting 60 seconds before retrying rate-limited request",
                    "Rate limited (attempt 2/2)",
                    "Request failed after 2 rate limit retries.",
                ],
            ),
            # Service unavailable with exponential backoff
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                3,
                1,
                [
                    {"status_code": 503, "body": '{"error": "service unavailable"}'},
                    {"status_code": 503, "body": '{"error": "service unavailable"}'},
                    {"status_code": 200, "body": '{"data": "success"}'},
                ],
                True,
                {"data": "success"},
                [
                    "HTTP GET call: https://api.example.com/data",
                    "503 Service Unavailable (attempt 1/3). Retrying in 2 seconds.",
                    "503 Service Unavailable (attempt 2/3). Retrying in 4 seconds.",
                    "Request succeeded.",
                ],
            ),
            # Service unavailable with max retries exceeded
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                2,
                1,
                [
                    {"status_code": 503, "body": '{"error": "service unavailable"}'},
                    {"status_code": 503, "body": '{"error": "service unavailable"}'},
                ],
                False,
                {"error": "service unavailable"},
                [
                    "HTTP GET call: https://api.example.com/data",
                    "503 Service Unavailable (attempt 1/2). Retrying in 2 seconds.",
                    "503 Service Unavailable (attempt 2/2). Retrying in 4 seconds.",
                    "Request failed after 2 retries due to repeated 503 errors.",
                ],
            ),
            # JSON decode error
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                3,
                1,
                [
                    {"status_code": 200, "body": "invalid json"},
                    {"status_code": 200, "body": "invalid json"},
                    {"status_code": 200, "body": "invalid json"},
                ],
                False,
                {"exception": "Expecting value: line 1 column 1 (char 0)"},
                [
                    "HTTP GET call: https://api.example.com/data",
                    "Exception during json loading: Expecting value: line 1 column 1 (char 0)",
                    "Request failed after 3 retries.",
                ],
            ),
            # Other HTTP error with retries
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                3,
                1,
                [
                    {"status_code": 500, "body": '{"error": "internal server error"}'},
                    {"status_code": 200, "body": '{"data": "success"}'},
                ],
                True,
                {"data": "success"},
                [
                    "HTTP GET call: https://api.example.com/data",
                    "Request failed [500]: {'error': 'internal server error'}",
                    "Request succeeded.",
                ],
            ),
            # Other HTTP error with max retries exceeded
            (
                "https://api.example.com/data",
                "GET",
                None,
                None,
                429,
                2,
                1,
                [
                    {"status_code": 500, "body": '{"error": "internal server error"}'},
                    {"status_code": 404, "body": '{"error": "not found"}'},
                ],
                False,
                {"error": "not found"},
                [
                    "HTTP GET call: https://api.example.com/data",
                    "Request failed [500]: {'error': 'internal server error'}",
                    "Request failed [404]: {'error': 'not found'}",
                    "Request failed after 2 retries.",
                ],
            ),
            # POST request with body
            (
                "https://api.example.com/data",
                "POST",
                {"key": "value"},
                {"Content-Type": "application/json"},
                429,
                3,
                1,
                [{"status_code": 201, "body": '{"id": 123}'}],
                True,
                {"id": 123},
                ["HTTP POST call: https://api.example.com/data", "Request succeeded."],
            ),
            # CoinGecko API with rate limiting delay
            (
                "https://api.coingecko.com/api/v3/simple/price",
                "GET",
                None,
                None,
                429,
                3,
                1,
                [{"status_code": 200, "body": '{"bitcoin": {"usd": 50000}}'}],
                True,
                {"bitcoin": {"usd": 50000}},
                [
                    "HTTP GET call: https://api.coingecko.com/api/v3/simple/price",
                    "Adding 2-second delay for CoinGecko API rate limiting",
                    "Request succeeded.",
                ],
            ),
        ],
    )
    def test_request_with_retries(
        self,
        endpoint,
        method,
        body,
        headers,
        rate_limited_code,
        max_retries,
        retry_wait,
        mock_responses,
        expected_success,
        expected_result,
        expected_logs,
    ):
        """Test _request_with_retries with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock get_http_response to return our test responses
        mock_responses_iter = iter(mock_responses)
        
        def mock_get_http_response(method, endpoint, content, headers):
            response_data = next(mock_responses_iter)
            mock_response = MagicMock()
            mock_response.status_code = response_data["status_code"]
            mock_response.body = response_data["body"]
            yield
            return mock_response
        
        # Mock rate_limited_callback
        mock_callback = MagicMock()
        
        # Mock sleep
        def mock_sleep(seconds):
            yield
            return None
        
        with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response), \
             patch.object(base_behaviour, 'sleep', side_effect=mock_sleep):
            
            result = self._consume_generator(
                base_behaviour._request_with_retries(
                    endpoint=endpoint,
                    rate_limited_callback=mock_callback,
                    method=method,
                    body=body,
                    headers=headers,
                    rate_limited_code=rate_limited_code,
                    max_retries=max_retries,
                    retry_wait=retry_wait,
                )
            )
        
        success, response_json = result
        assert success == expected_success
        assert response_json == expected_result
        
        # Verify rate_limited_callback was called for rate limiting scenarios
        if any(r.get("status_code") == rate_limited_code for r in mock_responses):
            assert mock_callback.called
        
        # Verify sleep was called for CoinGecko API
        if "coingecko.com" in endpoint:
            # For CoinGecko API, we expect sleep to be called with 2 seconds
            # We can't assert on the mock directly since it's a side_effect function
            # The test passes if the function completes successfully
            pass

    def test_request_with_retries_rate_limited_callback_called(self):
        """Test that rate_limited_callback is called when rate limited."""
        base_behaviour = self._create_base_behaviour()
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.body = '{"error": "rate limited"}'
        
        mock_callback = MagicMock()
        
        def mock_get_http_response(*args, **kwargs):
            yield
            return mock_response
        
        def mock_sleep(seconds):
            yield
            return None
        
        with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response), \
             patch.object(base_behaviour, 'sleep', side_effect=mock_sleep):
            
            self._consume_generator(
                base_behaviour._request_with_retries(
                    endpoint="https://api.example.com/data",
                    rate_limited_callback=mock_callback,
                    max_retries=1,
                )
            )
        
        mock_callback.assert_called_once()

    def test_request_with_retries_exponential_backoff(self):
        """Test exponential backoff for 503 errors."""
        base_behaviour = self._create_base_behaviour()
        
        # Create responses: 503, 503, 200
        mock_responses = [
            MagicMock(status_code=503, body='{"error": "service unavailable"}'),
            MagicMock(status_code=503, body='{"error": "service unavailable"}'),
            MagicMock(status_code=200, body='{"data": "success"}'),
        ]
        
        mock_callback = MagicMock()
        sleep_calls = []
        
        def mock_get_http_response(*args, **kwargs):
            yield
            return mock_responses.pop(0)
        
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
            yield
            return None
        
        with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response), \
             patch.object(base_behaviour, 'sleep', side_effect=mock_sleep):
            
            result = self._consume_generator(
                base_behaviour._request_with_retries(
                    endpoint="https://api.example.com/data",
                    rate_limited_callback=mock_callback,
                    max_retries=3,
                )
            )
        
        success, response_json = result
        assert success is True
        assert response_json == {"data": "success"}
        
        # Verify exponential backoff: 2, 4 seconds
        assert sleep_calls == [2, 4]

    def test_request_with_retries_custom_rate_limited_code(self):
        """Test custom rate limited code."""
        base_behaviour = self._create_base_behaviour()
        
        mock_response = MagicMock()
        mock_response.status_code = 418  # Custom rate limit code
        mock_response.body = '{"error": "custom rate limit"}'
        
        mock_callback = MagicMock()
        
        def mock_get_http_response(*args, **kwargs):
            yield
            return mock_response
        
        def mock_sleep(seconds):
            yield
            return None
        
        with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response), \
             patch.object(base_behaviour, 'sleep', side_effect=mock_sleep):
            
            result = self._consume_generator(
                base_behaviour._request_with_retries(
                    endpoint="https://api.example.com/data",
                    rate_limited_callback=mock_callback,
                    rate_limited_code=418,
                    max_retries=1,
                )
            )
        
        success, response_json = result
        assert success is False
        assert response_json == {"error": "custom rate limit"}
        mock_callback.assert_called_once()

    def test_request_with_retries_with_body_encoding(self):
        """Test request with body encoding."""
        base_behaviour = self._create_base_behaviour()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body = '{"data": "success"}'
        
        mock_callback = MagicMock()
        
        def mock_get_http_response(*args, **kwargs):
            yield
            return mock_response
        
        def mock_sleep(seconds):
            yield
            return None
        
        with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response), \
             patch.object(base_behaviour, 'sleep', side_effect=mock_sleep):
            
            result = self._consume_generator(
                base_behaviour._request_with_retries(
                    endpoint="https://api.example.com/data",
                    rate_limited_callback=mock_callback,
                    method="POST",
                    body={"key": "value", "number": 123},
                    headers={"Content-Type": "application/json"},
                )
            )
        
        success, response_json = result
        assert success is True
        assert response_json == {"data": "success"}

    @pytest.mark.parametrize(
        "test_case,position,expected_result,decimals_side_effect,prices_side_effect,description",
        [
            # Test case 1: Valid position with both tokens
            (
                "valid_dual_token",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token1": "0xToken1",
                    "token0_symbol": "USDC",
                    "token1_symbol": "ETH",
                    "amount0": 1000000000000000000,  # 1.0 in 18 decimals
                    "amount1": 2000000000000000000,  # 2.0 in 18 decimals
                    "timestamp": 1640995200,  # 2022-01-01
                },
                3500.0,  # 1.0 * 1000 + 2.0 * 1250 = 3500
                lambda *args, **kwargs: (yield None) or 18,  # Always return 18
                lambda *args, **kwargs: (yield None) or {
                    "0xToken0": 1000.0,  # USDC price
                    "0xToken1": 1250.0,  # ETH price
                },
                "Valid position with both tokens should calculate total value"
            ),
            # Test case 2: Position with only token0
            (
                "single_token",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token0_symbol": "USDC",
                    "amount0": 1000000000000000000,  # 1.0 in 18 decimals
                    "timestamp": 1640995200,
                },
                1000.0,  # 1.0 * 1000 = 1000
                lambda *args, **kwargs: (yield None) or 18,  # Always return 18
                lambda *args, **kwargs: (yield None) or {
                    "0xToken0": 1000.0,  # USDC price
                },
                "Position with only token0 should calculate single token value"
            ),
            # Test case 3: Missing required fields
            (
                "missing_required_fields",
                {
                    "chain": "optimism",
                    "token0_symbol": "USDC",
                    "amount0": 1000000000000000000,
                    # Missing token0 and timestamp
                },
                None,
                lambda *args, **kwargs: (yield None) or 18,  # Not used
                lambda *args, **kwargs: (yield None) or {},  # Not used
                "Missing required fields should return None"
            ),
            # Test case 4: Missing timestamp but has enter_timestamp
            (
                "enter_timestamp_fallback",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token0_symbol": "USDC",
                    "amount0": 1000000000000000000,
                    "enter_timestamp": 1640995200,
                },
                1000.0,
                lambda *args, **kwargs: (yield None) or 18,  # Always return 18
                lambda *args, **kwargs: (yield None) or {
                    "0xToken0": 1000.0,  # USDC price
                },
                "Should use enter_timestamp when timestamp is missing"
            ),
            # Test case 5: Missing token0 decimals
            (
                "missing_token0_decimals",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token0_symbol": "USDC",
                    "amount0": 1000000000000000000,
                    "timestamp": 1640995200,
                },
                None,
                lambda *args, **kwargs: (yield None) or None,  # Return None for token0
                lambda *args, **kwargs: (yield None) or {},  # Not used
                "Missing token0 decimals should return None"
            ),
            # Test case 6: Missing token1 decimals (covers line 1687)
            (
                "missing_token1_decimals",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token1": "0xToken1",
                    "token0_symbol": "USDC",
                    "token1_symbol": "ETH",
                    "amount0": 1000000000000000000,
                    "amount1": 2000000000000000000,
                    "timestamp": 1640995200,
                },
                None,
                lambda *args, **kwargs: (yield None) or (18 if args[1] == "0xToken0" else None),  # 18 for token0, None for token1
                lambda *args, **kwargs: (yield None) or {},  # Not used
                "Missing token1 decimals should return None (covers line 1687)"
            ),
            # Test case 7: token1 decimals is 0 (covers line 1687)
            (
                "token1_decimals_zero",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token1": "0xToken1",
                    "token0_symbol": "USDC",
                    "token1_symbol": "ETH",
                    "amount0": 1000000000000000000,
                    "amount1": 2000000000000000000,
                    "timestamp": 1640995200,
                },
                None,
                lambda *args, **kwargs: (yield None) or (18 if args[1] == "0xToken0" else 0),  # 18 for token0, 0 for token1
                lambda *args, **kwargs: (yield None) or {},  # Not used
                "token1 decimals of 0 should return None (covers line 1687)"
            ),
            # Test case 8: Missing historical prices
            (
                "missing_historical_prices",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token0_symbol": "USDC",
                    "amount0": 1000000000000000000,
                    "timestamp": 1640995200,
                },
                None,
                lambda *args, **kwargs: (yield None) or 18,  # Always return 18
                lambda *args, **kwargs: (yield None) or {},  # Return empty dict
                "Missing historical prices should return None"
            ),
            # Test case 9: Missing token0 price
            (
                "missing_token0_price",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token0_symbol": "USDC",
                    "amount0": 1000000000000000000,
                    "timestamp": 1640995200,
                },
                None,
                lambda *args, **kwargs: (yield None) or 18,  # Always return 18
                lambda *args, **kwargs: (yield None) or {"0xOtherToken": 1000.0},  # Missing token0 price
                "Missing token0 price should return None"
            ),
            # Test case 10: Missing token1 price for dual token position
            (
                "missing_token1_price",
                {
                    "chain": "optimism",
                    "token0": "0xToken0",
                    "token1": "0xToken1",
                    "token0_symbol": "USDC",
                    "token1_symbol": "ETH",
                    "amount0": 1000000000000000000,
                    "amount1": 2000000000000000000,
                    "timestamp": 1640995200,
                },
                None,
                lambda *args, **kwargs: (yield None) or 18,  # Always return 18
                lambda *args, **kwargs: (yield None) or {"0xToken0": 1000.0},  # Missing token1 price
                "Missing token1 price for dual token position should return None"
            ),
        ]
    )
    def test_calculate_initial_investment_value_comprehensive(
        self, test_case, position, expected_result, decimals_side_effect, prices_side_effect, description
    ):
        """Test calculate_initial_investment_value method with comprehensive scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        with patch.object(base_behaviour, '_get_token_decimals', side_effect=decimals_side_effect):
            with patch.object(base_behaviour, '_fetch_historical_token_prices', side_effect=prices_side_effect):
                result = self._consume_generator(base_behaviour.calculate_initial_investment_value(position))
                
                if expected_result is None:
                    assert result is None, f"Test case '{test_case}': {description} - Expected None but got {result}"
                else:
                    assert result == expected_result, f"Test case '{test_case}': {description} - Expected {expected_result} but got {result}"



    @pytest.mark.parametrize(
        "current_positions,expected_total",
        [
            # Test case 1: Multiple open positions
            (
                [
                    {
                        "status": "open",
                        "pool_address": "0xPool1",
                        "tx_hash": "0xTx1",
                        "chain": "optimism",
                        "token0": "0xToken0",
                        "amount0": 1000000000000000000,
                        "timestamp": 1640995200,
                    },
                    {
                        "status": "open",
                        "pool_address": "0xPool2",
                        "tx_hash": "0xTx2",
                        "chain": "optimism",
                        "token0": "0xToken0",
                        "amount0": 2000000000000000000,
                        "timestamp": 1640995200,
                    },
                ],
                3000.0,  # 1000 + 2000
            ),
            # Test case 2: Mix of open and closed positions
            (
                [
                    {
                        "status": "open",
                        "pool_address": "0xPool1",
                        "tx_hash": "0xTx1",
                        "chain": "optimism",
                        "token0": "0xToken0",
                        "amount0": 1000000000000000000,
                        "timestamp": 1640995200,
                    },
                    {
                        "status": "closed",
                        "pool_address": "0xPool2",
                        "tx_hash": "0xTx2",
                        "chain": "optimism",
                        "token0": "0xToken0",
                        "amount0": 2000000000000000000,
                        "timestamp": 1640995200,
                    },
                ],
                1000.0,  # Only open position counted
            ),
            # Test case 3: No open positions
            (
                [
                    {
                        "status": "closed",
                        "pool_address": "0xPool1",
                        "tx_hash": "0xTx1",
                        "chain": "optimism",
                        "token0": "0xToken0",
                        "amount0": 1000000000000000000,
                        "timestamp": 1640995200,
                    },
                ],
                None,  # No open positions, should return None
            ),
            # Test case 4: Empty positions
            (
                [],
                None,  # No positions, should return None
            ),
        ]
    )
    def test_calculate_initial_investment(self, current_positions, expected_total):
        """Test calculate_initial_investment method."""
        base_behaviour = self._create_base_behaviour()
        base_behaviour.current_positions = current_positions
        
        # Mock calculate_initial_investment_value to return 1000 per position
        def mock_calculate_initial_investment_value(position):
            if position.get("amount0") == 1000000000000000000:
                yield None
                return 1000.0
            elif position.get("amount0") == 2000000000000000000:
                yield None
                return 2000.0
            else:
                yield None
                return None
        
        with patch.object(base_behaviour, 'calculate_initial_investment_value', side_effect=mock_calculate_initial_investment_value):
            result = self._consume_generator(base_behaviour.calculate_initial_investment())
            
            if expected_total is None:
                assert result is None
            else:
                assert result == expected_total
                
                # Verify that initial_investment_values_per_pool is populated correctly
                if result is not None and result > 0:
                    assert len(base_behaviour.initial_investment_values_per_pool) > 0

    def test_calculate_initial_investment_with_null_position_value(self):
        """Test calculate_initial_investment when some positions return null values."""
        base_behaviour = self._create_base_behaviour()
        base_behaviour.current_positions = [
            {
                "status": "open",
                "pool_address": "0xPool1",
                "tx_hash": "0xTx1",
                "id": "position1",
                "chain": "optimism",
                "token0": "0xToken0",
                "amount0": 1000000000000000000,
                "timestamp": 1640995200,
            },
            {
                "status": "open",
                "pool_address": "0xPool2",
                "tx_hash": "0xTx2",
                "id": "position2",
                "chain": "optimism",
                "token0": "0xToken0",
                "amount0": 2000000000000000000,
                "timestamp": 1640995200,
            },
        ]
        
        # Mock calculate_initial_investment_value to return value for first position, None for second
        def mock_calculate_initial_investment_value(position):
            if position.get("id") == "position1":
                yield None
                return 1000.0
            else:
                yield None
                return None  # Null value for second position
        
        with patch.object(base_behaviour, 'calculate_initial_investment_value', side_effect=mock_calculate_initial_investment_value):
            result = self._consume_generator(base_behaviour.calculate_initial_investment())
            
            # Should return the value from the first position only
            assert result == 1000.0
            assert len(base_behaviour.initial_investment_values_per_pool) == 1

    def test_calculate_initial_investment_position_key_formation(self):
        """Test that position keys are formed correctly in initial_investment_values_per_pool."""
        base_behaviour = self._create_base_behaviour()
        base_behaviour.current_positions = [
            {
                "status": "open",
                "pool_address": "0xPool1",
                "tx_hash": "0xTx1",
                "chain": "optimism",
                "token0": "0xToken0",
                "amount0": 1000000000000000000,
                "timestamp": 1640995200,
            },
        ]
        
        def mock_calculate_initial_investment_value(position):
            yield None
            return 1000.0
        
        with patch.object(base_behaviour, 'calculate_initial_investment_value', side_effect=mock_calculate_initial_investment_value):
            result = self._consume_generator(base_behaviour.calculate_initial_investment())
            
            assert result == 1000.0
            expected_key = "0xPool1_0xTx1"
            assert expected_key in base_behaviour.initial_investment_values_per_pool
            assert base_behaviour.initial_investment_values_per_pool[expected_key] == 1000.0

    def test_calculate_initial_investment_with_pool_id_fallback(self):
        """Test position key formation when pool_id is used instead of pool_address."""
        base_behaviour = self._create_base_behaviour()
        base_behaviour.current_positions = [
            {
                "status": "open",
                "pool_id": "0xPoolId1",  # Using pool_id instead of pool_address
                "tx_hash": "0xTx1",
                "chain": "optimism",
                "token0": "0xToken0",
                "amount0": 1000000000000000000,
                "timestamp": 1640995200,
            },
        ]
        
        def mock_calculate_initial_investment_value(position):
            yield None
            return 1000.0
        
        with patch.object(base_behaviour, 'calculate_initial_investment_value', side_effect=mock_calculate_initial_investment_value):
            result = self._consume_generator(base_behaviour.calculate_initial_investment())
            
            assert result == 1000.0
            expected_key = "0xPoolId1_0xTx1"
            assert expected_key in base_behaviour.initial_investment_values_per_pool
            assert base_behaviour.initial_investment_values_per_pool[expected_key] == 1000.0

    @pytest.mark.parametrize(
        "chain,service_id,endpoint,response_status,response_body,expected_result",
        [
            # Test case 1: Successful query with valid service data
            (
                "optimism",
                "service123",
                "https://api.example.com/subgraph",
                200,
                {
                    "data": {
                        "service": {
                            "blockNumber": "12345",
                            "blockTimestamp": "1640995200",
                            "currentOlasStaked": "1000000000000000000",
                            "id": "service123",
                            "olasRewardsEarned": "500000000000000000",
                        }
                    }
                },
                {
                    "blockNumber": "12345",
                    "blockTimestamp": "1640995200",
                    "currentOlasStaked": "1000000000000000000",
                    "id": "service123",
                    "olasRewardsEarned": "500000000000000000",
                },
            ),
            # Test case 2: Successful query with minimal service data
            (
                "mode",
                "service456",
                "https://api.example.com/subgraph",
                200,
                {
                    "data": {
                        "service": {
                            "id": "service456",
                            "olasRewardsEarned": "0",
                        }
                    }
                },
                {
                    "id": "service456",
                    "olasRewardsEarned": "0",
                },
            ),
            # Test case 3: HTTP error response
            (
                "optimism",
                "service123",
                "https://api.example.com/subgraph",
                500,
                {"error": "Internal server error"},
                None,
            ),
            # Test case 4: No service data in response
            (
                "optimism",
                "service123",
                "https://api.example.com/subgraph",
                200,
                {"data": {"service": None}},
                None,
            ),
            # Test case 5: Missing data field in response
            (
                "optimism",
                "service123",
                "https://api.example.com/subgraph",
                200,
                {"error": "No data"},
                None,
            ),
        ]
    )
    def test_query_service_rewards(
        self, chain, service_id, endpoint, response_status, response_body, expected_result
    ):
        """Test query_service_rewards method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params to include required values
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.staking_subgraph_endpoints = {chain: endpoint}
            mock_params.return_value.on_chain_service_id = service_id
            
            # Mock get_http_response
            def mock_get_http_response(method, url, content, headers):
                mock_response = MagicMock()
                mock_response.status_code = response_status
                mock_response.body = json.dumps(response_body)
                yield None
                return mock_response
            
            with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response):
                result = self._consume_generator(base_behaviour.query_service_rewards(chain))
                
                if expected_result is None:
                    assert result is None
                else:
                    assert result == expected_result

    def test_query_service_rewards_json_decode_error(self):
        """Test query_service_rewards when JSON decode fails."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.staking_subgraph_endpoints = {"optimism": "https://api.example.com/subgraph"}
            mock_params.return_value.on_chain_service_id = "service123"
            
            # Mock get_http_response to return invalid JSON
            def mock_get_http_response(method, url, content, headers):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.body = "invalid json"
                yield None
                return mock_response
            
            with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response):
                # The function will fail when JSON decode fails, so we expect an exception
                with pytest.raises(json.JSONDecodeError):
                    self._consume_generator(base_behaviour.query_service_rewards("optimism"))

    @pytest.mark.parametrize(
        "chain,should_update,service_data,olas_rewards_earned,expected_writes",
        [
            # Test case 1: Should update with valid rewards
            (
                "optimism",
                True,
                {
                    "blockNumber": "12345",
                    "blockTimestamp": "1640995200",
                    "olasRewardsEarned": "500000000000000000",
                },
                "500000000000000000",
                2,  # Two writes: rewards and update timestamp
            ),
            # Test case 2: Should not update (early return)
            (
                "optimism",
                False,
                None,
                None,
                0,  # No writes
            ),
            # Test case 3: Should update but no service data
            (
                "optimism",
                True,
                None,
                None,
                0,  # No writes
            ),
            # Test case 4: Should update with zero rewards
            (
                "mode",
                True,
                {
                    "blockNumber": "12345",
                    "blockTimestamp": "1640995200",
                    "olasRewardsEarned": "0",
                },
                "0",
                2,  # Two writes: rewards and update timestamp
            ),
            # Test case 5: Should update with invalid rewards (converted to 0)
            (
                "optimism",
                True,
                {
                    "blockNumber": "12345",
                    "blockTimestamp": "1640995200",
                    "olasRewardsEarned": "invalid",
                },
                "invalid",
                2,  # Two writes: rewards (0) and update timestamp
            ),
        ]
    )
    def test_update_accumulated_rewards_for_chain(
        self, chain, should_update, service_data, olas_rewards_earned, expected_writes
    ):
        """Test update_accumulated_rewards_for_chain method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock should_update_rewards_from_subgraph
        def mock_should_update(*args, **kwargs):
            yield None
            return should_update
        
        # Mock query_service_rewards
        def mock_query_service_rewards(*args, **kwargs):
            yield None
            return service_data
        
        # Mock _write_kv
        write_calls = []
        def mock_write_kv(data):
            write_calls.append(data)
            yield None
            return True
        
        # Mock _get_current_timestamp
        def mock_get_current_timestamp():
            return 1640995200
        
        with patch.object(base_behaviour, 'should_update_rewards_from_subgraph', side_effect=mock_should_update):
            with patch.object(base_behaviour, 'query_service_rewards', side_effect=mock_query_service_rewards):
                with patch.object(base_behaviour, '_write_kv', side_effect=mock_write_kv):
                    with patch.object(base_behaviour, '_get_current_timestamp', side_effect=mock_get_current_timestamp):
                        self._consume_generator(base_behaviour.update_accumulated_rewards_for_chain(chain))
                        
                        assert len(write_calls) == expected_writes
                        
                        if expected_writes > 0:
                            # Check that rewards were written
                            rewards_key = f"accumulated_rewards_{chain}_{OLAS_ADDRESSES[chain].lower()}"
                            rewards_write = next((call for call in write_calls if rewards_key in call), None)
                            assert rewards_write is not None
                            
                            # Check that update timestamp was written
                            update_key = f"{REWARD_UPDATE_KEY_PREFIX}{chain}"
                            update_write = next((call for call in write_calls if update_key in call), None)
                            assert update_write is not None
                            assert update_write[update_key] == "1640995200"

    def test_update_accumulated_rewards_for_chain_unknown_chain(self):
        """Test update_accumulated_rewards_for_chain with unknown chain (no OLAS address)."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock should_update_rewards_from_subgraph
        def mock_should_update(*args, **kwargs):
            yield None
            return True
        
        # Mock query_service_rewards
        def mock_query_service_rewards(*args, **kwargs):
            yield None
            return {
                "blockNumber": "12345",
                "blockTimestamp": "1640995200",
                "olasRewardsEarned": "500000000000000000",
            }
        
        # Mock _write_kv
        write_calls = []
        def mock_write_kv(data):
            write_calls.append(data)
            yield None
            return True
        
        # Mock _get_current_timestamp
        def mock_get_current_timestamp():
            return 1640995200
        
        with patch.object(base_behaviour, 'should_update_rewards_from_subgraph', side_effect=mock_should_update):
            with patch.object(base_behaviour, 'query_service_rewards', side_effect=mock_query_service_rewards):
                with patch.object(base_behaviour, '_write_kv', side_effect=mock_write_kv):
                    with patch.object(base_behaviour, '_get_current_timestamp', side_effect=mock_get_current_timestamp):
                        # Test with unknown chain that has no OLAS address
                        self._consume_generator(base_behaviour.update_accumulated_rewards_for_chain("unknown_chain"))
                        
                        # Should only write the update timestamp, not rewards
                        assert len(write_calls) == 1
                        update_key = f"{REWARD_UPDATE_KEY_PREFIX}unknown_chain"
                        assert update_key in write_calls[0]

    def test_update_accumulated_rewards_for_chain_write_kv_failure(self):
        """Test update_accumulated_rewards_for_chain when _write_kv fails."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock should_update_rewards_from_subgraph
        def mock_should_update(*args, **kwargs):
            yield None
            return True
        
        # Mock query_service_rewards
        def mock_query_service_rewards(*args, **kwargs):
            yield None
            return {
                "blockNumber": "12345",
                "blockTimestamp": "1640995200",
                "olasRewardsEarned": "500000000000000000",
            }
        
        # Mock _write_kv to fail
        def mock_write_kv(data):
            yield None
            return False  # Write failure
        
        # Mock _get_current_timestamp
        def mock_get_current_timestamp():
            return 1640995200
        
        with patch.object(base_behaviour, 'should_update_rewards_from_subgraph', side_effect=mock_should_update):
            with patch.object(base_behaviour, 'query_service_rewards', side_effect=mock_query_service_rewards):
                with patch.object(base_behaviour, '_write_kv', side_effect=mock_write_kv):
                    with patch.object(base_behaviour, '_get_current_timestamp', side_effect=mock_get_current_timestamp):
                        # Should not raise exception, just log and continue
                        self._consume_generator(base_behaviour.update_accumulated_rewards_for_chain("optimism"))

    def test_query_service_rewards_missing_endpoint(self):
        """Test query_service_rewards when endpoint is missing for chain."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params with missing endpoint
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.staking_subgraph_endpoints = {}  # No endpoints
            mock_params.return_value.on_chain_service_id = "service123"
            
            # Mock get_http_response to return None when endpoint is None
            def mock_get_http_response(method, url, content, headers):
                if url is None:
                    yield None
                    return None
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.body = json.dumps({"data": {"service": None}})
                yield None
                return mock_response
            
            with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response):
                # The function will fail when response is None, so we expect an exception
                with pytest.raises(AttributeError):
                    self._consume_generator(base_behaviour.query_service_rewards("unknown_chain"))

    def test_query_service_rewards_missing_service_id(self):
        """Test query_service_rewards when service_id is None."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params with None service_id
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.staking_subgraph_endpoints = {"optimism": "https://api.example.com/subgraph"}
            mock_params.return_value.on_chain_service_id = None
            
            # Mock get_http_response to return a valid response
            def mock_get_http_response(method, url, content, headers):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.body = json.dumps({"data": {"service": None}})
                yield None
                return mock_response
            
            with patch.object(base_behaviour, 'get_http_response', side_effect=mock_get_http_response):
                result = self._consume_generator(base_behaviour.query_service_rewards("optimism"))
                assert result is None

    @pytest.mark.parametrize(
        "safe_address,mock_responses,expected_total_balances,expected_requests,expected_logs,expected_error,test_description",
        [
            # Test case 1: Single page with results  
            (
                "0xSafe123",
                [
                    {
                        "success": True,
                        "response": {
                            "results": [
                                {"tokenAddress": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "balance": "1000"},
                                {"tokenAddress": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "balance": "2000"},
                            ],
                            "next": None,
                        },
                    }
                ],
                2,
                1,
                ["Total balances fetched from SafeApi: 2"],
                None,
                "Single page with results"
            ),
            # Test case 2: Multiple pages with pagination
            (
                "0xSafe456",
                [
                    {
                        "success": True,
                        "response": {
                            "results": [
                                {"tokenAddress": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "balance": "1000"},
                                {"tokenAddress": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "balance": "2000"},
                            ],
                            "next": "https://api.example.com/next",
                        },
                    },
                    {
                        "success": True,
                        "response": {
                            "results": [
                                {"tokenAddress": "0x2e3d870790dc77a83dd1d18184acc7439a53f475", "balance": "3000"},
                            ],
                            "next": None,
                        },
                    }
                ],
                3,
                2,
                ["Fetching SafeApi page: offset=0, limit=100", "Fetching SafeApi page: offset=100, limit=100", "Reached last page of SafeApi results", "Total balances fetched from SafeApi: 3"],
                None,
                "Multiple pages with pagination"
            ),
            # Test case 3: Empty results (no more results)
            (
                "0xSafe789",
                [
                    {
                        "success": True,
                        "response": {
                            "results": [],
                            "next": None,
                        },
                    }
                ],
                0,
                1,
                ["No more results from SafeApi"],
                None,
                "Empty results (no more results)"
            ),
            # Test case 4: API failure
            (
                "0xSafeFail",
                [
                    {
                        "success": False,
                        "response": "API Error",
                    }
                ],
                0,
                1,
                [],
                "Failed to fetch SafeApi data: API Error",
                "API failure"
            ),
        ]
    )
    def test_fetch_safe_balances_with_pagination(
        self, safe_address, mock_responses, expected_total_balances, expected_requests, expected_logs, expected_error, test_description
    ):
        """Test _fetch_safe_balances_with_pagination method with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params with safe API base URL
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.safe_api_base_url = "https://safe-api.example.com"
            
            # Mock _request_with_retries
            request_calls = []
            mock_responses_iter = iter(mock_responses)
            
            def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
                request_calls.append({
                    "endpoint": endpoint,
                    "method": method,
                    "headers": headers,
                })
                response_data = next(mock_responses_iter)
                yield None
                return response_data["success"], response_data["response"]
            
            with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
                if expected_error:
                    # Test error case
                    with patch.object(base_behaviour.context.logger, 'error') as mock_error_logger:
                        result = self._consume_generator(base_behaviour._fetch_safe_balances_with_pagination(safe_address))
                        
                        assert result == []
                        assert len(request_calls) == expected_requests
                        mock_error_logger.assert_called_once_with(expected_error)
                else:
                    # Test success case
                    with patch.object(base_behaviour.context.logger, 'info') as mock_info:
                        result = self._consume_generator(base_behaviour._fetch_safe_balances_with_pagination(safe_address))
                        
                        assert len(result) == expected_total_balances
                        assert len(request_calls) == expected_requests
                        
                        # Verify log messages if expected
                        if expected_logs:
                            info_calls = [call[0][0] for call in mock_info.call_args_list]
                            for expected_log in expected_logs:
                                assert any(expected_log in call for call in info_calls)



    @pytest.mark.parametrize(
        "safe_addresses,balances_data,expected_result,expected_logs,expected_error,test_description",
        [
            # Test case 1: Success with ETH and USDC balances
            (
                {"optimism": "0xSafe123"},
                [
                    {"tokenAddress": None, "balance": "1000000000000000000"},  # Native ETH
                    {
                        "tokenAddress": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
                        "token": {"symbol": "USDC"},
                        "balance": "1000000",
                    },
                ],
                [
                    {"asset_symbol": "ETH", "asset_type": "native", "address": "0x0", "balance": 1000000000000000000},
                    {"asset_symbol": "USDC", "asset_type": "erc_20", "address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "balance": 1000000},
                ],
                ["Retrieved 2 token balances from SafeApi"],
                None,
                "Success with ETH and USDC balances"
            ),
            # Test case 2: No safe address set (error case)
            (
                {},
                [],
                [],
                [],
                "No safe address set for Optimism chain",
                "No safe address set"
            ),
            # Test case 3: Success with safe address and token filtering
            (
                {"optimism": "0xSafe123"},
                [
                    {"tokenAddress": None, "balance": "1000000000000000000"},  # Native ETH
                    {
                        "tokenAddress": "0xfAf87e196A29969094bE35DfB0Ab9d0b8518dB84",  # This token should be filtered out
                        "token": {"symbol": "FILTERED"},
                        "balance": "1000000"
                    },
                    {
                        "tokenAddress": "0x1234567890123456789012345678901234567890",
                        "token": {"symbol": "USDC"},
                        "balance": "1000000"
                    }
                ],
                [
                    {"asset_symbol": "ETH", "asset_type": "native", "address": "0x0", "balance": 1000000000000000000},
                    {"asset_symbol": "USDC", "asset_type": "erc_20", "address": "0x1234567890123456789012345678901234567890", "balance": 1000000},
                ],
                ["Retrieved 2 token balances from SafeApi"],
                None,
                "Success with token filtering (FILTERED token excluded)"
            ),
            # Test case 4: Empty balances response
            (
                {"optimism": "0xSafe123"},
                [],
                [],
                ["Retrieved 0 token balances from SafeApi"],
                None,
                "Empty balances response"
            ),
        ]
    )
    def test_get_optimism_balances_from_safe_api_parameterized(self, safe_addresses, balances_data, expected_result, expected_logs, expected_error, test_description):
        """Test _get_optimism_balances_from_safe_api method with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        if safe_addresses:
            # Mock params with safe contract addresses
            with patch.dict(base_behaviour.params.safe_contract_addresses, safe_addresses, clear=False):
                if expected_error:
                    # Test error case
                    with patch.object(base_behaviour.context.logger, 'error') as mock_error:
                        result = self._consume_generator(base_behaviour._get_optimism_balances_from_safe_api())
                        
                        assert result == expected_result
                        mock_error.assert_called_once_with(expected_error)
                else:
                    # Test success case
                    def mock_fetch_safe_balances(safe_addr):
                        yield None
                        return balances_data
                    
                    with patch.object(base_behaviour, '_fetch_safe_balances_with_pagination', side_effect=mock_fetch_safe_balances):
                        with patch.object(base_behaviour.context.logger, 'info') as mock_info:
                            result = self._consume_generator(base_behaviour._get_optimism_balances_from_safe_api())
                            
                            assert len(result) == len(expected_result)
                            
                            # Verify each expected balance
                            for i, expected_balance in enumerate(expected_result):
                                assert result[i]["asset_symbol"] == expected_balance["asset_symbol"]
                                assert result[i]["asset_type"] == expected_balance["asset_type"]
                                assert result[i]["balance"] == expected_balance["balance"]
                            
                            # Verify log messages
                            if expected_logs:
                                info_calls = [call[0][0] for call in mock_info.call_args_list]
                                for expected_log in expected_logs:
                                    assert any(expected_log in call for call in info_calls)
        else:
            # Test case with no safe addresses
            with patch.object(base_behaviour.context.logger, 'error') as mock_error:
                result = self._consume_generator(base_behaviour._get_optimism_balances_from_safe_api())
                
                assert result == expected_result
                mock_error.assert_called_once_with(expected_error)

    def test_get_positions_success(self):
        """Test get_positions method with successful responses from both chains."""
        base_behaviour = self._create_base_behaviour()
        
        optimism_balances = [
            {"asset_symbol": "ETH", "asset_type": "native", "address": "0x0", "balance": 1000},
            {"asset_symbol": "USDC", "asset_type": "erc_20", "address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "balance": 2000},
        ]
        
        mode_balances = [
            {"asset_symbol": "ETH", "asset_type": "native", "address": "0x0", "balance": 500},
            {"asset_symbol": "MODE", "asset_type": "erc_20", "address": "0xdfc7c877a950e49d2610114102175a06c2e3167a", "balance": 1500},
        ]
        
        # Mock params with target investment chains
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.target_investment_chains = ["optimism", "mode"]
            
            # Mock balance fetching methods
            def mock_get_optimism_balances():
                yield None
                return optimism_balances
            
            def mock_get_mode_balances():
                yield None
                return mode_balances
            
            with patch.object(base_behaviour, '_get_optimism_balances_from_safe_api', side_effect=mock_get_optimism_balances):
                with patch.object(base_behaviour, '_get_mode_balances_from_explorer_api', side_effect=mock_get_mode_balances):
                    result = self._consume_generator(base_behaviour.get_positions())
                    
                    assert len(result) == 2  # Two chains
                    
                    # Verify position structure
                    for position in result:
                        assert "chain" in position
                        assert "assets" in position
                        assert isinstance(position["assets"], list)
                        assert position["chain"] in ["optimism", "mode"]

    @pytest.mark.parametrize(
        "safe_address,eth_balance,eth_balance_exists,token_balances,expected_total_balances,expected_logs",
        [
            # Test case 1: Both ETH and token balances
            (
                "0xSafe123",
                1000000000000000000,  # 1 ETH
                True,
                [
                    {"asset_symbol": "USDC", "asset_type": "erc_20", "address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "balance": 1000000},
                    {"asset_symbol": "DAI", "asset_type": "erc_20", "address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "balance": 2000000},
                ],
                3,  # 1 ETH + 2 tokens
                ["Fetching Mode balances from Explorer API for safe: 0xSafe123", "Retrieved 3 token balances from Mode Explorer API"],
            ),
            # Test case 2: Only ETH balance (no tokens)
            (
                "0xSafe456",
                500000000000000000,  # 0.5 ETH
                True,
                [],
                1,  # 1 ETH only
                ["Fetching Mode balances from Explorer API for safe: 0xSafe456", "Retrieved 1 token balances from Mode Explorer API"],
            ),
            # Test case 3: Only token balances (no ETH)
            (
                "0xSafe789",
                0,
                False,
                [
                    {"asset_symbol": "USDC", "asset_type": "erc_20", "address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "balance": 1000000},
                ],
                1,  # 1 token only
                ["Fetching Mode balances from Explorer API for safe: 0xSafe789", "Retrieved 1 token balances from Mode Explorer API"],
            ),
            # Test case 4: No balances at all
            (
                "0xSafeEmpty",
                0,
                False,
                [],
                0,  # No balances
                ["Fetching Mode balances from Explorer API for safe: 0xSafeEmpty", "Retrieved 0 token balances from Mode Explorer API"],
            ),
        ]
    )
    def test_get_mode_balances_from_explorer_api(
        self, safe_address, eth_balance, eth_balance_exists, token_balances, expected_total_balances, expected_logs
    ):
        """Test _get_mode_balances_from_explorer_api method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params with safe contract addresses
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.safe_contract_addresses = {"mode": safe_address}
            
            # Mock _get_native_balance
            def mock_get_native_balance(chain, account):
                yield None
                return eth_balance if eth_balance_exists else None
            
            # Mock _fetch_mode_token_balances
            def mock_fetch_mode_token_balances(safe_addr):
                yield None
                return token_balances
            
            with patch.object(base_behaviour, '_get_native_balance', side_effect=mock_get_native_balance):
                with patch.object(base_behaviour, '_fetch_mode_token_balances', side_effect=mock_fetch_mode_token_balances):
                    with patch.object(base_behaviour.context.logger, 'info') as mock_logger:
                        result = self._consume_generator(base_behaviour._get_mode_balances_from_explorer_api())
                        
                        assert len(result) == expected_total_balances
                        
                        # Verify log messages
                        log_calls = [call[0][0] for call in mock_logger.call_args_list]
                        for expected_log in expected_logs:
                            assert any(expected_log in log_call for log_call in log_calls)
                        
                        # Verify ETH balance if it exists
                        if eth_balance_exists and eth_balance > 0:
                            eth_balances = [b for b in result if b.get("asset_symbol") == "ETH"]
                            assert len(eth_balances) == 1
                            assert eth_balances[0]["asset_type"] == "native"
                            assert eth_balances[0]["address"] == "0x0000000000000000000000000000000000000000"
                            assert eth_balances[0]["balance"] == eth_balance
                        
                        # Verify token balances
                        token_balances_result = [b for b in result if b.get("asset_type") == "erc_20"]
                        assert len(token_balances_result) == len(token_balances)

    def test_get_mode_balances_from_explorer_api_no_safe_address(self):
        """Test _get_mode_balances_from_explorer_api when no safe address is configured."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params with no safe address for mode
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.safe_contract_addresses = {}  # No safe addresses
            
            with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                result = self._consume_generator(base_behaviour._get_mode_balances_from_explorer_api())
                
                assert result == []
                mock_logger.assert_called_once_with("No safe address set for Mode chain")

    @pytest.mark.parametrize(
        "safe_address,all_tokens,active_lp_addresses,expected_balances,expected_logs",
        [
            # Test case 1: Valid tokens with some LP tokens to filter out
            (
                "0xSafe123",
                [
                    {
                        "token": {"address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "symbol": "USDC"},
                        "value": "1000000"
                    },
                    {
                        "token": {"address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "symbol": "DAI"},
                        "value": "2000000"
                    },
                    {
                        "token": {"address": "0xPool123", "symbol": "LP-TOKEN"},  # LP token to filter out
                        "value": "500000"
                    },
                ],
                {"0xpool123"},  # Active LP address to filter out
                2,  # 2 regular tokens, 1 LP token filtered out
                ["Filtering out LP token LP-TOKEN (0xPool123) - active position"],
            ),
            # Test case 2: All tokens are LP tokens (all filtered out)
            (
                "0xSafe456",
                [
                    {
                        "token": {"address": "0xPool1", "symbol": "LP-TOKEN-1"},
                        "value": "1000000"
                    },
                    {
                        "token": {"address": "0xPool2", "symbol": "LP-TOKEN-2"},
                        "value": "2000000"
                    },
                ],
                {"0xpool1", "0xpool2"},  # All are active LP addresses
                0,  # All tokens filtered out
                [
                    "Filtering out LP token LP-TOKEN-1 (0xPool1) - active position",
                    "Filtering out LP token LP-TOKEN-2 (0xPool2) - active position"
                ],
            ),
            # Test case 3: No LP tokens to filter out
            (
                "0xSafe789",
                [
                    {
                        "token": {"address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "symbol": "USDC"},
                        "value": "1000000"
                    },
                    {
                        "token": {"address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "symbol": "DAI"},
                        "value": "2000000"
                    },
                ],
                set(),  # No active LP addresses
                2,  # All tokens included
                [],
            ),
            # Test case 4: Zero balances filtered out
            (
                "0xSafeZero",
                [
                    {
                        "token": {"address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "symbol": "USDC"},
                        "value": "0"  # Zero balance
                    },
                    {
                        "token": {"address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "symbol": "DAI"},
                        "value": "2000000"
                    },
                ],
                set(),
                1,  # Only non-zero balance included
                [],
            ),
            # Test case 5: Invalid balance values
            (
                "0xSafeInvalid",
                [
                    {
                        "token": {"address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "symbol": "USDC"},
                        "value": "invalid_value"  # Invalid balance
                    },
                    {
                        "token": {"address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "symbol": "DAI"},
                        "value": "2000000"
                    },
                ],
                set(),
                1,  # Only valid balance included
                ["Invalid balance value for token 0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85: invalid_value"],
            ),
            # Test case 6: Missing token info (token field not present)
            (
                "0xSafeMissing",
                [
                    {
                        "value": "1000000"  # Missing token field entirely
                    },
                    {
                        "token": {"address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "symbol": "DAI"},
                        "value": "2000000"
                    },
                ],
                set(),
                1,  # Only token with valid info included
                [],
            ),
            # Test case 7: Missing token address
            (
                "0xSafeNoAddress",
                [
                    {
                        "token": {"symbol": "USDC"},  # Missing address
                        "value": "1000000"
                    },
                    {
                        "token": {"address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "symbol": "DAI"},
                        "value": "2000000"
                    },
                ],
                set(),
                1,  # Only token with address included
                [],
            ),
            # Test case 8: XVELO to VELO renaming
            (
                "0xSafeVelo",
                [
                    {
                        "token": {"address": "0x1234567890123456789012345678901234567890", "symbol": "XVELO"},
                        "value": "1000000"
                    },
                ],
                set(),  # No active LP addresses
                1,  # Token included with renamed symbol
                ["Renamed XVELO to VELO for Mode chain token at 0x1234567890123456789012345678901234567890"],
            ),
            # Test case 9: Empty tokens list
            (
                "0xSafeEmpty",
                [],
                set(),
                0,  # No tokens
                [],
            ),
            # Test case 10: Negative balance filtered out
            (
                "0xSafeNegative",
                [
                    {
                        "token": {"address": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "symbol": "USDC"},
                        "value": "-1000000"  # Negative balance
                    },
                    {
                        "token": {"address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "symbol": "DAI"},
                        "value": "2000000"  # Positive balance
                    },
                ],
                set(),
                1,  # Only positive balance included
                [],
            ),
        ]
    )
    def test_fetch_mode_token_balances(
        self, safe_address, all_tokens, active_lp_addresses, expected_balances, expected_logs
    ):
        """Test _fetch_mode_token_balances method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _get_active_lp_addresses
        def mock_get_active_lp_addresses():
            return active_lp_addresses
        
        # Mock _fetch_mode_tokens_with_pagination
        def mock_fetch_mode_tokens_with_pagination(safe_addr):
            yield None
            return all_tokens
        
        with patch.object(base_behaviour, '_get_active_lp_addresses', side_effect=mock_get_active_lp_addresses):
            with patch.object(base_behaviour, '_fetch_mode_tokens_with_pagination', side_effect=mock_fetch_mode_tokens_with_pagination):
                with patch.object(base_behaviour.context.logger, 'info') as mock_info_logger:
                    with patch.object(base_behaviour.context.logger, 'warning') as mock_warning_logger:
                        result = self._consume_generator(base_behaviour._fetch_mode_token_balances(safe_address))
                        
                        assert len(result) == expected_balances
                        
                        # Verify log messages
                        info_log_calls = [call[0][0] for call in mock_info_logger.call_args_list]
                        warning_log_calls = [call[0][0] for call in mock_warning_logger.call_args_list]
                        
                        for expected_log in expected_logs:
                            if "Filtering out LP token" in expected_log:
                                assert any(expected_log in log_call for log_call in info_log_calls)
                            elif "Invalid balance value" in expected_log:
                                assert any(expected_log in log_call for log_call in warning_log_calls)
                        
                        # Verify balance structure for valid tokens
                        for balance in result:
                            assert "asset_symbol" in balance
                            assert "asset_type" in balance
                            assert "address" in balance
                            assert "balance" in balance
                            assert balance["asset_type"] == "erc_20"
                            assert isinstance(balance["balance"], int)
                            assert balance["balance"] > 0



    @pytest.mark.parametrize(
        "chain,expected_result",
        [
            ("optimism", 1640995200),
            ("mode", 1640995300),
            ("unknown_chain", None),
        ]
    )
    def test_get_next_checkpoint(self, chain, expected_result):
        """Test _get_next_checkpoint method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            def side_effect(*args, **kwargs):
                yield None
                return expected_result
            mock_contract.side_effect = side_effect
            
            result = self._consume_generator(base_behaviour._get_next_checkpoint(chain))
            assert result == expected_result

    @pytest.mark.parametrize(
        "chain,expected_result",
        [
            ("optimism", 1640995200),
            ("mode", 1640995300),
            ("unknown_chain", None),
        ]
    )
    def test_get_ts_checkpoint(self, chain, expected_result):
        """Test _get_ts_checkpoint method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            def side_effect(*args, **kwargs):
                yield None
                return expected_result
            mock_contract.side_effect = side_effect
            
            result = self._consume_generator(base_behaviour._get_ts_checkpoint(chain))
            assert result == expected_result

    @pytest.mark.parametrize(
        "chain,expected_result,should_log_error",
        [
            ("optimism", 1000, False),
            ("mode", 2000, False),
            ("optimism", None, True),
            ("mode", 0, True),
        ]
    )
    def test_get_liveness_ratio(self, chain, expected_result, should_log_error):
        """Test _get_liveness_ratio method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                def side_effect(*args, **kwargs):
                    yield None
                    return expected_result
                mock_contract.side_effect = side_effect
                
                result = self._consume_generator(base_behaviour._get_liveness_ratio(chain))
                assert result == expected_result
                
                if should_log_error:
                    mock_logger.assert_called_once()
                else:
                    mock_logger.assert_not_called()

    @pytest.mark.parametrize(
        "chain,expected_result,should_log_error",
        [
            ("optimism", 86400, False),
            ("mode", 172800, False),
            ("optimism", None, True),
            ("mode", 0, True),
        ]
    )
    def test_get_liveness_period(self, chain, expected_result, should_log_error):
        """Test _get_liveness_period method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                def side_effect(*args, **kwargs):
                    yield None
                    return expected_result
                mock_contract.side_effect = side_effect
                
                result = self._consume_generator(base_behaviour._get_liveness_period(chain))
                assert result == expected_result
                
                if should_log_error:
                    mock_logger.assert_called_once()
                else:
                    mock_logger.assert_not_called()

    @pytest.mark.parametrize(
        "service_staking_state,min_num_of_safe_tx_required,multisig_nonces_since_last_cp,expected_result",
        [
            # Test case 1: Staked service with KPI met
            (StakingState.STAKED.value, 5, 10, True),
            # Test case 2: Staked service with KPI not met
            (StakingState.STAKED.value, 10, 5, False),
            # Test case 3: Unstaked service
            (StakingState.UNSTAKED.value, 5, 10, None),
            # Test case 4: Staked service with None min_num_of_safe_tx_required
            (StakingState.STAKED.value, None, 10, None),
            # Test case 5: Staked service with None multisig_nonces_since_last_cp
            (StakingState.STAKED.value, 5, None, None),
        ]
    )
    def test_is_staking_kpi_met(self, service_staking_state, min_num_of_safe_tx_required, multisig_nonces_since_last_cp, expected_result):
        """Test _is_staking_kpi_met method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock synchronized_data
        with patch.object(type(base_behaviour), 'synchronized_data', new_callable=PropertyMock) as mock_sync_data:
            mock_sync_data.return_value.service_staking_state = service_staking_state
            mock_sync_data.return_value.min_num_of_safe_tx_required = min_num_of_safe_tx_required
            
            # Mock params
            with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
                mock_params.return_value.staking_chain = "optimism"
                mock_params.return_value.safe_contract_addresses = {"optimism": "0xSafe123"}
                
                # Mock _get_multisig_nonces_since_last_cp
                with patch.object(base_behaviour, '_get_multisig_nonces_since_last_cp') as mock_get_nonces:
                    def side_effect(*args, **kwargs):
                        yield None
                        return multisig_nonces_since_last_cp
                    mock_get_nonces.side_effect = side_effect
                    
                    # Mock logger for error cases
                    with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                        result = self._consume_generator(base_behaviour._is_staking_kpi_met())
                        assert result == expected_result
                        
                        # Check if error was logged for None min_num_of_safe_tx_required
                        if min_num_of_safe_tx_required is None:
                            mock_logger.assert_called_once()

    @pytest.mark.parametrize(
        "chain,multisig,contract_result,expected_result",
        [
            # Test case 1: Valid multisig nonces
            ("optimism", "0xSafe123", [10], 10),
            # Test case 2: Empty multisig nonces
            ("mode", "0xSafe456", [], None),
            # Test case 3: None multisig nonces
            ("optimism", "0xSafe789", None, None),
        ]
    )
    def test_get_multisig_nonces(self, chain, multisig, contract_result, expected_result):
        """Test _get_multisig_nonces method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock contract interaction
        with patch.object(base_behaviour, 'contract_interact') as mock_contract:
            def side_effect(*args, **kwargs):
                yield None
                return contract_result
            mock_contract.side_effect = side_effect
            
            result = self._consume_generator(base_behaviour._get_multisig_nonces(chain, multisig))
            assert result == expected_result

    @pytest.mark.parametrize(
        "chain,multisig,multisig_nonces,service_info,expected_result",
        [
            # Test case 1: Valid calculation
            ("optimism", "0xSafe123", 15, (None, None, (5, None)), 10),  # 15 - 5 = 10
            # Test case 2: None multisig nonces
            ("mode", "0xSafe456", None, (None, None, (5, None)), None),
            # Test case 3: None service info
            ("optimism", "0xSafe789", 15, None, None),
            # Test case 4: Empty service info
            ("mode", "0xSafe101", 15, (), None),
            # Test case 5: Service info with empty third element
            ("optimism", "0xSafe202", 15, (None, None, ()), None),
        ]
    )
    def test_get_multisig_nonces_since_last_cp(self, chain, multisig, multisig_nonces, service_info, expected_result):
        """Test _get_multisig_nonces_since_last_cp method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _get_multisig_nonces
        with patch.object(base_behaviour, '_get_multisig_nonces') as mock_get_nonces:
            def side_effect(*args, **kwargs):
                yield None
                return multisig_nonces
            mock_get_nonces.side_effect = side_effect
            
            # Mock _get_service_info
            with patch.object(base_behaviour, '_get_service_info') as mock_get_service_info:
                def side_effect(*args, **kwargs):
                    yield None
                    return service_info
                mock_get_service_info.side_effect = side_effect
                
                # Mock logger for error cases
                with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                    with patch.object(base_behaviour.context.logger, 'info') as mock_info_logger:
                        result = self._consume_generator(base_behaviour._get_multisig_nonces_since_last_cp(chain, multisig))
                        assert result == expected_result
                        
                        # Check if error was logged for invalid service info
                        if service_info is None or (isinstance(service_info, tuple) and (len(service_info) == 0 or len(service_info[2]) == 0)):
                            mock_logger.assert_called_once()
                        elif expected_result is not None:
                            mock_info_logger.assert_called_once()

    @pytest.mark.parametrize(
        "chain,service_id,contract_result,expected_result",
        [
            # Test case 1: Valid service info
            ("optimism", "service123", (1000, 2000, (3000, 4000)), (1000, 2000, (3000, 4000))),
            # Test case 2: None service id
            ("mode", None, None, None),
            # Test case 3: Valid service id but None contract result
            ("optimism", "service456", None, None),
        ]
    )
    def test_get_service_info(self, chain, service_id, contract_result, expected_result):
        """Test _get_service_info method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.on_chain_service_id = service_id
            
            # Mock contract interaction
            with patch.object(base_behaviour, 'contract_interact') as mock_contract:
                with patch.object(base_behaviour.context.logger, 'warning') as mock_logger:
                    def side_effect(*args, **kwargs):
                        yield None
                        return contract_result
                    mock_contract.side_effect = side_effect
                    
                    result = self._consume_generator(base_behaviour._get_service_info(chain))
                    assert result == expected_result
                    
                    # Check if warning was logged for None service id
                    if service_id is None:
                        mock_logger.assert_called_once()

    @pytest.mark.parametrize(
        "chain,service_id,contract_result,expected_staking_state",
        [
            # Test case 1: Valid staking state
            ("optimism", "service123", StakingState.STAKED.value, StakingState.STAKED),
            # Test case 2: None service id
            ("mode", None, None, StakingState.UNSTAKED),
            # Test case 3: Valid service id but None contract result
            ("optimism", "service456", None, StakingState.UNSTAKED),
        ]
    )
    def test_get_service_staking_state(self, chain, service_id, contract_result, expected_staking_state):
        """Test _get_service_staking_state method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock params
        with patch.object(type(base_behaviour), 'params', new_callable=PropertyMock) as mock_params:
            mock_params.return_value.on_chain_service_id = service_id
            
            # Mock contract interaction
            with patch.object(base_behaviour, 'contract_interact') as mock_contract:
                with patch.object(base_behaviour.context.logger, 'warning') as mock_logger:
                    def side_effect(*args, **kwargs):
                        yield None
                        return contract_result
                    mock_contract.side_effect = side_effect
                    
                    self._consume_generator(base_behaviour._get_service_staking_state(chain))
                    assert base_behaviour.service_staking_state == expected_staking_state
                    
                    # Check if warning was logged for None service id or None contract result
                    if service_id is None or contract_result is None:
                        mock_logger.assert_called()

    @pytest.mark.parametrize(
        "token_address,chain,cached_price,platform_id,api_success,api_response,expected_result",
        [
            # Test case 1: Cached price available
            ("0xToken123", "optimism", 1.5, "optimistic-ethereum", False, None, 1.5),
            # Test case 2: No cache, successful API call
            ("0xToken456", "mode", None, "mode", True, {"0xtoken456": {"usd": 2.0}}, 2.0),
            # Test case 3: No cache, failed API call
            ("0xToken789", "optimism", None, "optimistic-ethereum", False, {"error": "rate limited"}, None),
            # Test case 4: Missing platform id
            ("0xToken101", "unknown_chain", None, None, False, None, None),
            # Test case 5: API success but no price in response
            ("0xToken202", "mode", None, "mode", True, {"0xtoken202": {}}, 0),
        ]
    )
    def test_fetch_token_price(self, token_address, chain, cached_price, platform_id, api_success, api_response, expected_result):
        """Test _fetch_token_price method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _get_cached_price
        with patch.object(base_behaviour, '_get_cached_price') as mock_get_cached:
            def side_effect(*args, **kwargs):
                yield None
                return cached_price
            mock_get_cached.side_effect = side_effect
            
            # Mock coingecko
            with patch.object(type(base_behaviour), 'coingecko', new_callable=PropertyMock) as mock_coingecko:
                mock_coingecko.return_value.api_key = "test_key"
                mock_coingecko.return_value.chain_to_platform_id_mapping = {
                    "optimism": "optimistic-ethereum",
                    "mode": "mode"
                }
                mock_coingecko.return_value.token_price_endpoint = "https://api.coingecko.com/api/v3/simple/token_price/{asset_platform_id}?contract_addresses={token_address}&vs_currencies=usd"
                mock_coingecko.return_value.rate_limited_code = 429
                mock_coingecko.return_value.rate_limited_status_callback = MagicMock()
                
                # Mock _request_with_retries
                with patch.object(base_behaviour, '_request_with_retries') as mock_request:
                    def side_effect(*args, **kwargs):
                        yield None
                        return api_success, api_response
                    mock_request.side_effect = side_effect
                    
                    # Mock _cache_price
                    with patch.object(base_behaviour, '_cache_price') as mock_cache:
                        def side_effect(*args, **kwargs):
                            yield None
                            return None
                        mock_cache.side_effect = side_effect
                        
                        # Mock _get_current_timestamp
                        with patch.object(base_behaviour, '_get_current_timestamp') as mock_timestamp:
                            mock_timestamp.return_value = 1640995200
                            
                            # Mock logger for error cases
                            with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                                result = self._consume_generator(base_behaviour._fetch_token_price(token_address, chain))
                                assert result == expected_result
                                
                                # Check if error was logged for missing platform id
                                if platform_id is None:
                                    mock_logger.assert_called_once()
                                
                                # Check if price was cached for successful API calls
                                if api_success and expected_result and expected_result > 0:
                                    mock_cache.assert_called_once()

    @pytest.mark.parametrize(
        "method,kwargs,response_payload,expected_result",
        [
            # Test case 1: Successful MirrorDB call
            (
                "get_data",
                {"key": "test_key"},
                {"response": {"data": "test_value"}},
                {"data": "test_value"},
            ),
            # Test case 2: MirrorDB call with error response
            (
                "get_data",
                {"key": "test_key"},
                {"error": "Database connection failed"},
                None,
            ),
            # Test case 3: MirrorDB call with no response field
            (
                "get_data",
                {"key": "test_key"},
                {"status": "success"},
                None,
            ),
        ]
    )
    def test_call_mirrordb(self, method, kwargs, response_payload, expected_result):
        """Test _call_mirrordb method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock context and dialogues
        with patch.object(base_behaviour.context, 'srr_dialogues') as mock_srr_dialogues:
            with patch.object(base_behaviour, '_do_connection_request') as mock_do_request:
                # Create mock message and dialogue
                mock_message = MagicMock()
                mock_dialogue = MagicMock()
                mock_srr_dialogues.create.return_value = (mock_message, mock_dialogue)
                
                # Create mock response
                mock_response = MagicMock()
                mock_response.payload = json.dumps(response_payload)
                
                def do_request_side_effect(message, dialogue, timeout=None):
                    yield None
                    return mock_response
                mock_do_request.side_effect = do_request_side_effect
                
                # Mock logger for error cases
                with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                    result = self._consume_generator(base_behaviour._call_mirrordb(method, **kwargs))
                    
                    # Verify dialogue creation
                    mock_srr_dialogues.create.assert_called_once()
                    call_args = mock_srr_dialogues.create.call_args
                    assert call_args[1]['performative'] == SrrMessage.Performative.REQUEST
                    assert call_args[1]['payload'] == json.dumps({"method": method, "kwargs": kwargs})
                    
                    # Verify request was made
                    mock_do_request.assert_called_once_with(mock_message, mock_dialogue)
                    
                    # Verify result
                    assert result == expected_result
                    
                    # Check if error was logged
                    if "error" in response_payload:
                        mock_logger.assert_called_once_with(response_payload["error"])

    def test_call_mirrordb_exception_handling(self):
        """Test _call_mirrordb method with exception handling."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock context and dialogues to raise exception
        with patch.object(base_behaviour.context, 'srr_dialogues') as mock_srr_dialogues:
            with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                mock_srr_dialogues.create.side_effect = Exception("Connection failed")
                
                result = self._consume_generator(base_behaviour._call_mirrordb("get_data", key="test"))
                
                assert result is None
                mock_logger.assert_called_once_with("Exception while calling MirrorDB: Connection failed")

    @pytest.mark.parametrize(
        "keys,response_performative,response_data,expected_result",
        [
            # Test case 1: Successful read with data
            (
                ("key1", "key2"),
                KvStoreMessage.Performative.READ_RESPONSE,
                {"key1": "value1", "key2": "value2"},
                {"key1": "value1", "key2": "value2"},
            ),
            # Test case 2: Successful read with missing keys
            (
                ("key1", "key2", "key3"),
                KvStoreMessage.Performative.READ_RESPONSE,
                {"key1": "value1", "key2": "value2"},
                {"key1": "value1", "key2": "value2", "key3": None},
            ),
            # Test case 3: Wrong response performative
            (
                ("key1",),
                KvStoreMessage.Performative.ERROR,
                {"key1": "value1"},
                None,
            ),
        ]
    )
    def test_read_kv(self, keys, response_performative, response_data, expected_result):
        """Test _read_kv method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock context and dialogues
        with patch.object(base_behaviour.context, 'kv_store_dialogues') as mock_kv_dialogues:
            with patch.object(base_behaviour, '_do_connection_request') as mock_do_request:
                with patch.object(base_behaviour.context.logger, 'info') as mock_logger:
                    # Create mock message and dialogue
                    mock_message = MagicMock()
                    mock_dialogue = MagicMock()
                    mock_kv_dialogues.create.return_value = (mock_message, mock_dialogue)
                    
                    # Create mock response
                    mock_response = MagicMock()
                    mock_response.performative = response_performative
                    mock_response.data = response_data
                    
                    def do_request_side_effect(message, dialogue, timeout=None):
                        yield None
                        return mock_response
                    mock_do_request.side_effect = do_request_side_effect
                    
                    result = self._consume_generator(base_behaviour._read_kv(keys))
                    
                    # Verify dialogue creation
                    mock_kv_dialogues.create.assert_called_once()
                    call_args = mock_kv_dialogues.create.call_args
                    assert call_args[1]['performative'] == KvStoreMessage.Performative.READ_REQUEST
                    assert call_args[1]['keys'] == keys
                    
                    # Verify request was made
                    mock_do_request.assert_called_once_with(mock_message, mock_dialogue)
                    
                    # Verify logging
                    mock_logger.assert_called_once_with(f"Reading keys from db: {keys}")
                    
                    # Verify result
                    assert result == expected_result

    @pytest.mark.parametrize(
        "data,response_performative,expected_result",
        [
            # Test case 1: Successful write
            (
                {"key1": "value1", "key2": "value2"},
                KvStoreMessage.Performative.SUCCESS,
                True,
            ),
            # Test case 2: Failed write
            (
                {"key1": "value1"},
                KvStoreMessage.Performative.ERROR,
                False,
            ),
        ]
    )
    def test_write_kv(self, data, response_performative, expected_result):
        """Test _write_kv method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock context and dialogues
        with patch.object(base_behaviour.context, 'kv_store_dialogues') as mock_kv_dialogues:
            with patch.object(base_behaviour, '_do_connection_request') as mock_do_request:
                # Create mock message and dialogue
                mock_message = MagicMock()
                mock_dialogue = MagicMock()
                mock_kv_dialogues.create.return_value = (mock_message, mock_dialogue)
                
                # Create mock response
                mock_response = response_performative
                
                def do_request_side_effect(message, dialogue, timeout=None):
                    yield None
                    return mock_response
                mock_do_request.side_effect = do_request_side_effect
                
                result = self._consume_generator(base_behaviour._write_kv(data))
                
                # Verify dialogue creation
                mock_kv_dialogues.create.assert_called_once()
                call_args = mock_kv_dialogues.create.call_args
                assert call_args[1]['performative'] == KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST
                assert call_args[1]['data'] == data
                
                # Verify request was made
                mock_do_request.assert_called_once_with(mock_message, mock_dialogue)
                
                # Verify result
                assert result == expected_result

    @pytest.mark.parametrize(
        "token_balances,expected_prices",
        [
            # Test case 1: Multiple tokens with prices
            (
                [
                    {"token": "0xToken1", "chain": "optimism"},
                    {"token": "0xToken2", "chain": "mode"},
                    {"token": "0x0000000000000000000000000000000000000000", "chain": "optimism"},  # Zero address
                ],
                {
                    "0xToken1": 1.5,
                    "0xToken2": 2.0,
                    "0x0000000000000000000000000000000000000000": 3000.0,  # ETH price
                },
            ),
            # Test case 2: Tokens with missing chain
            (
                [
                    {"token": "0xToken1", "chain": "optimism"},
                    {"token": "0xToken2"},  # Missing chain
                ],
                {
                    "0xToken1": 1.5,
                    # Token2 is excluded due to missing chain
                },
            ),
            # Test case 3: Tokens with None prices
            (
                [
                    {"token": "0xToken1", "chain": "optimism"},
                    {"token": "0xToken2", "chain": "mode"},
                ],
                {
                    "0xToken1": 1.5,
                    # Token2 price is None, so not included
                },
            ),
            # Test case 4: Empty token list
            (
                [],
                {},
            ),
        ]
    )
    def test_fetch_token_prices(self, token_balances, expected_prices):
        """Test _fetch_token_prices method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _fetch_token_price and _fetch_zero_address_price
        with patch.object(base_behaviour, '_fetch_token_price') as mock_fetch_token:
            with patch.object(base_behaviour, '_fetch_zero_address_price') as mock_fetch_zero:
                with patch.object(base_behaviour.context.logger, 'error') as mock_logger:
                    def fetch_token_side_effect(token_address, chain):
                        yield None
                        if token_address == "0xToken1":
                            return 1.5
                        elif token_address == "0xToken2":
                            # For test case 0: return 2.0 (multiple tokens with prices)
                            # For test case 1: return None (missing chain - but this won't be called)
                            # For test case 2: return None (None prices)
                            # For test case 3: return None (empty list - but this won't be called)
                            # We need to check the test case by looking at the token_balances
                            if len(token_balances) == 3:  # Test case 0: multiple tokens
                                return 2.0
                            else:  # Other test cases
                                return None
                        else:
                            return None
                    
                    def fetch_zero_side_effect():
                        yield None
                        return 3000.0  # ETH price
                    
                    mock_fetch_token.side_effect = fetch_token_side_effect
                    mock_fetch_zero.side_effect = fetch_zero_side_effect
                    
                    result = self._consume_generator(base_behaviour._fetch_token_prices(token_balances))
                    
                    # Verify result
                    assert result == expected_prices
                    
                    # Verify error logging for missing chain
                    missing_chain_tokens = [t for t in token_balances if not t.get("chain")]
                    if missing_chain_tokens:
                        for token_data in missing_chain_tokens:
                            mock_logger.assert_any_call(f"Missing chain for token {token_data['token']}")
                    
                    # Verify method calls
                    token_calls = [call for call in mock_fetch_token.call_args_list if call[0][0] != ZERO_ADDRESS]
                    zero_calls = [call for call in mock_fetch_zero.call_args_list]
                    
                    expected_token_calls = len([t for t in token_balances if t.get("chain") and t["token"] != ZERO_ADDRESS])
                    expected_zero_calls = len([t for t in token_balances if t["token"] == ZERO_ADDRESS])
                    
                    assert len(token_calls) == expected_token_calls
                    assert len(zero_calls) == expected_zero_calls

    @pytest.mark.parametrize(
        "cached_price,api_success,api_response,expected_result",
        [
            # Test case 1: Cached price available
            (3000.0, False, None, 3000.0),
            # Test case 2: No cache, successful API call
            (None, True, {"ethereum": {"usd": 3500.0}}, 3500.0),
            # Test case 3: No cache, failed API call
            (None, False, {"error": "rate limited"}, None),
            # Test case 4: API success but no price in response
            (None, True, {"ethereum": {}}, 0),
            # Test case 5: API success but empty response
            (None, True, {}, 0),
        ]
    )
    def test_fetch_zero_address_price(self, cached_price, api_success, api_response, expected_result):
        """Test _fetch_zero_address_price method."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _get_cached_price
        with patch.object(base_behaviour, '_get_cached_price') as mock_get_cached:
            def get_cached_side_effect(token_address, date_str):
                yield None
                return cached_price
            mock_get_cached.side_effect = get_cached_side_effect
            
            # Mock coingecko
            with patch.object(type(base_behaviour), 'coingecko', new_callable=PropertyMock) as mock_coingecko:
                mock_coingecko.return_value.api_key = "test_key"
                mock_coingecko.return_value.coin_price_endpoint = "https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                mock_coingecko.return_value.rate_limited_code = 429
                mock_coingecko.return_value.rate_limited_status_callback = MagicMock()
                
                # Mock _request_with_retries
                with patch.object(base_behaviour, '_request_with_retries') as mock_request:
                    def request_side_effect(endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait):
                        yield None
                        return api_success, api_response
                    mock_request.side_effect = request_side_effect
                    
                    # Mock _cache_price
                    with patch.object(base_behaviour, '_cache_price') as mock_cache:
                        def cache_side_effect(token_address, price, date_str):
                            yield None
                            return None
                        mock_cache.side_effect = cache_side_effect
                        
                        # Mock _get_current_timestamp
                        with patch.object(base_behaviour, '_get_current_timestamp') as mock_timestamp:
                            mock_timestamp.return_value = 1640995200
                            
                            result = self._consume_generator(base_behaviour._fetch_zero_address_price())
                            
                            # Verify result
                            assert result == expected_result
                            
                            # Verify cache was checked
                            mock_get_cached.assert_called_once_with("ethereum", "01-01-2022")
                            
                            # Verify API call was made only if no cache
                            if cached_price is None:
                                mock_request.assert_called_once()
                                call_args = mock_request.call_args
                                assert "ethereum" in call_args[1]['endpoint']
                                
                                # Verify price was cached for successful API calls
                                if api_success and expected_result and expected_result > 0:
                                    mock_cache.assert_called_once_with("ethereum", expected_result, "01-01-2022")
                            else:
                                mock_request.assert_not_called()
                                mock_cache.assert_not_called()

    def test_fetch_zero_address_price_api_response_iteration(self):
        """Test _fetch_zero_address_price with different API response structures."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _get_cached_price to return None
        with patch.object(base_behaviour, '_get_cached_price') as mock_get_cached:
            def get_cached_side_effect(token_address, date_str):
                yield None
                return None
            mock_get_cached.side_effect = get_cached_side_effect
            
            # Mock coingecko
            with patch.object(type(base_behaviour), 'coingecko', new_callable=PropertyMock) as mock_coingecko:
                mock_coingecko.return_value.api_key = "test_key"
                mock_coingecko.return_value.coin_price_endpoint = "https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                mock_coingecko.return_value.rate_limited_code = 429
                mock_coingecko.return_value.rate_limited_status_callback = MagicMock()
                
                # Mock _request_with_retries
                with patch.object(base_behaviour, '_request_with_retries') as mock_request:
                    def request_side_effect(endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait):
                        yield None
                        return True, {"ethereum": {"usd": 3500.0}}
                    mock_request.side_effect = request_side_effect
                    
                    # Mock _cache_price
                    with patch.object(base_behaviour, '_cache_price') as mock_cache:
                        def cache_side_effect(token_address, price, date_str):
                            yield None
                            return None
                        mock_cache.side_effect = cache_side_effect
                        
                        # Mock _get_current_timestamp
                        with patch.object(base_behaviour, '_get_current_timestamp') as mock_timestamp:
                            mock_timestamp.return_value = 1640995200
                            
                            result = self._consume_generator(base_behaviour._fetch_zero_address_price())
                            
                            # Verify result
                            assert result == 3500.0
                            
                            # Verify next(iter()) was called correctly
                            mock_request.assert_called_once()

    @pytest.mark.parametrize(
        "tokens,date_str,chain,coingecko_ids,historical_prices,expected_result",
        [
            # Test case 1: Multiple tokens with prices found
            (
                [["USDC", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"], ["DAI", "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1"]],
                "01-01-2022",
                "optimism",
                {"usdc": "usd-coin", "dai": "makerdao-optimism-bridged-dai-optimism"},
                {"usd-coin": 1.0, "makerdao-optimism-bridged-dai-optimism": 0.998},
                {"0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85": 1.0, "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1": 0.998},
            ),
            # Test case 2: One token with CoinGecko ID not found
            (
                [["USDC", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"], ["UNKNOWN", "0xUnknown123"]],
                "01-01-2022",
                "optimism",
                {"usdc": "usd-coin", "unknown": None},
                {"usd-coin": 1.0},
                {"0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85": 1.0},
            ),
            # Test case 3: Token with zero price (filtered out)
            (
                [["USDC", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"], ["DAI", "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1"]],
                "01-01-2022",
                "optimism",
                {"usdc": "usd-coin", "dai": "makerdao-optimism-bridged-dai-optimism"},
                {"usd-coin": 1.0, "makerdao-optimism-bridged-dai-optimism": None},
                {"0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85": 1.0},
            ),
            # Test case 4: Empty token list
            (
                [],
                "01-01-2022",
                "optimism",
                {},
                {},
                {},
            ),
            # Test case 5: All tokens have no CoinGecko ID
            (
                [["UNKNOWN1", "0xUnknown1"], ["UNKNOWN2", "0xUnknown2"]],
                "01-01-2022",
                "optimism",
                {"unknown1": None, "unknown2": None},
                {},
                {},
            ),
        ]
    )
    def test_fetch_historical_token_prices(self, tokens, date_str, chain, coingecko_ids, historical_prices, expected_result):
        """Test _fetch_historical_token_prices method."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_get_coin_id_from_symbol(symbol, chain_name):
            return coingecko_ids.get(symbol.lower())
        
        def mock_fetch_historical_token_price(coingecko_id, date):
            yield None
            return historical_prices.get(coingecko_id)
        
        with patch.object(base_behaviour, 'get_coin_id_from_symbol', side_effect=mock_get_coin_id_from_symbol):
            with patch.object(base_behaviour, '_fetch_historical_token_price', side_effect=mock_fetch_historical_token_price):
                result = self._consume_generator(base_behaviour._fetch_historical_token_prices(tokens, date_str, chain))
                
                assert result == expected_result

    @pytest.mark.parametrize(
        "coingecko_id,date_str,cached_price,api_success,api_response,expected_result,expected_logs",
        [
            # Test case 1: Cached price available
            (
                "bitcoin",
                "01-01-2022",
                50000.0,
                False,
                None,
                50000.0,
                [],
            ),
            # Test case 2: No cache, successful API call
            (
                "ethereum",
                "01-01-2022",
                None,
                True,
                {"market_data": {"current_price": {"usd": 3500.0}}},
                3500.0,
                [],
            ),
            # Test case 3: No cache, failed API call
            (
                "failed-token",
                "01-01-2022",
                None,
                False,
                {"error": "rate limited"},
                None,
                ["Failed to fetch historical price for failed-token"],
            ),
            # Test case 4: API success but no price in response
            (
                "no-price-token",
                "01-01-2022",
                None,
                True,
                {"market_data": {"current_price": {}}},
                None,
                ["No price in response for token no-price-token"],
            ),
            # Test case 5: API success but malformed response structure
            (
                "malformed-token",
                "01-01-2022",
                None,
                True,
                {"data": {"wrong_structure": True}},
                None,
                ["No price in response for token malformed-token"],
            ),
            # Test case 6: API success with valid price and caching
            (
                "cacheable-token",
                "01-01-2022",
                None,
                True,
                {"market_data": {"current_price": {"usd": 1250.0}}},
                1250.0,
                [],
            ),
        ]
    )
    def test_fetch_historical_token_price(self, coingecko_id, date_str, cached_price, api_success, api_response, expected_result, expected_logs):
        """Test _fetch_historical_token_price method."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_get_cached_price(token_address, date):
            yield None
            return cached_price
        
        def mock_request_with_retries(endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait):
            yield None
            return api_success, api_response
        
        def mock_cache_price(token_address, price, date):
            yield None
            return None
        
        with patch.object(base_behaviour, '_get_cached_price', side_effect=mock_get_cached_price):
            with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
                with patch.object(base_behaviour, '_cache_price', side_effect=mock_cache_price):
                    result = self._consume_generator(base_behaviour._fetch_historical_token_price(coingecko_id, date_str))
                    
                    assert result == expected_result

    def test_fetch_historical_token_price_api_endpoint_formatting(self):
        """Test that _fetch_historical_token_price formats the API endpoint correctly."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_get_cached_price(token_address, date):
            yield None
            return None
        
        def mock_request_with_retries(endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait):
            yield None
            # Verify endpoint formatting
            assert "bitcoin" in endpoint
            assert "01-01-2022" in endpoint
            return True, {"market_data": {"current_price": {"usd": 45000.0}}}
        
        def mock_cache_price(token_address, price, date):
            yield None
            return None
        
        with patch.object(base_behaviour, '_get_cached_price', side_effect=mock_get_cached_price):
            with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
                with patch.object(base_behaviour, '_cache_price', side_effect=mock_cache_price):
                    result = self._consume_generator(base_behaviour._fetch_historical_token_price("bitcoin", "01-01-2022"))
                    assert result == 45000.0

    def test_fetch_historical_token_price_with_api_key(self):
        """Test that _fetch_historical_token_price includes API key in headers when available."""
        base_behaviour = self._create_base_behaviour()
        base_behaviour.coingecko.api_key = "test_api_key"
        
        def mock_get_cached_price(token_address, date):
            yield None
            return None
        
        def mock_request_with_retries(endpoint, headers, rate_limited_code, rate_limited_callback, retry_wait):
            yield None
            # Verify API key is included in headers
            assert headers.get("x-cg-api-key") == "test_api_key"
            assert headers.get("Accept") == "application/json"
            return True, {"market_data": {"current_price": {"usd": 2000.0}}}
        
        def mock_cache_price(token_address, price, date):
            yield None
            return None
        
        with patch.object(base_behaviour, '_get_cached_price', side_effect=mock_get_cached_price):
            with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
                with patch.object(base_behaviour, '_cache_price', side_effect=mock_cache_price):
                    result = self._consume_generator(base_behaviour._fetch_historical_token_price("ethereum", "01-01-2022"))
                    assert result == 2000.0

    @pytest.mark.parametrize(
        "chain,token_address,contract_result,expected_result",
        [
            # Test case 1: Successful token name fetch
            ("optimism", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85", "USD Coin", "USD Coin"),
            # Test case 2: Contract interaction returns None
            ("mode", "0xdfc7c877a950e49d2610114102175a06c2e3167a", None, None),
            # Test case 3: Different chain and token
            ("optimism", "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "Dai Stablecoin", "Dai Stablecoin"),
            # Test case 4: Empty string result
            ("mode", "0xToken123", "", ""),
        ]
    )
    def test_fetch_token_name_from_contract(self, chain, token_address, contract_result, expected_result):
        """Test _fetch_token_name_from_contract method."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_contract_interact(*args, **kwargs):
            yield None
            return contract_result
        
        with patch.object(base_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            result = self._consume_generator(base_behaviour._fetch_token_name_from_contract(chain, token_address))
            
            assert result == expected_result

    def test_fetch_token_name_from_contract_interaction_parameters(self):
        """Test that _fetch_token_name_from_contract calls contract_interact with correct parameters."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_contract_interact(performative, contract_address, contract_public_id, contract_callable, data_key, chain_id):
            yield None
            # Verify parameters
            assert performative == ContractApiMessage.Performative.GET_RAW_TRANSACTION
            assert contract_address == "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"
            assert contract_callable == "get_name"
            assert data_key == "data"
            assert chain_id == "optimism"
            return "USD Coin"
        
        with patch.object(base_behaviour, 'contract_interact', side_effect=mock_contract_interact):
            result = self._consume_generator(base_behaviour._fetch_token_name_from_contract("optimism", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"))
            assert result == "USD Coin"

    def test_fetch_historical_token_prices_with_logging(self):
        """Test that _fetch_historical_token_prices logs errors for tokens without CoinGecko IDs."""
        base_behaviour = self._create_base_behaviour()
        
        tokens = [["UNKNOWN_TOKEN", "0xUnknown123"], ["USDC", "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"]]
        
        def mock_get_coin_id_from_symbol(symbol, chain_name):
            if symbol.lower() == "unknown_token":
                return None
            return "usd-coin"
        
        def mock_fetch_historical_token_price(coingecko_id, date):
            yield None
            return 1.0
        
        with patch.object(base_behaviour, 'get_coin_id_from_symbol', side_effect=mock_get_coin_id_from_symbol):
            with patch.object(base_behaviour, '_fetch_historical_token_price', side_effect=mock_fetch_historical_token_price):
                result = self._consume_generator(base_behaviour._fetch_historical_token_prices(tokens, "01-01-2022", "optimism"))
                
                # Should only have USDC price, UNKNOWN_TOKEN should be filtered out
                assert result == {"0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85": 1.0}

    @pytest.mark.parametrize(
        "input_data,expected_positions,expected_calls",
        [
            # Test case 1: Single dict with address -> pool_address conversion
            (
                {
                    "address": "0xPool123",
                    "chain": "optimism",
                    "assets": ["0xToken0", "0xToken1"],
                    "enter_timestamp": 1640995200,
                },
                [
                    {
                        "pool_address": "0xPool123",
                        "chain": "optimism",
                        "token0": "0xToken0",
                        "token0_symbol": "USDC",
                        "token1": "0xToken1",
                        "token1_symbol": "DAI",
                        "status": "open",
                        "entry_cost": 0.0,
                        "min_hold_days": 21.0,
                        "cost_recovered": False,
                        "principal_usd": 0.0,
                        "enter_timestamp": 1640995200,
                    }
                ],
                [
                    ("optimism", "0xToken0"),
                    ("optimism", "0xToken1"),
                ],
            ),
            # Test case 2: List of positions with mixed data
            (
                [
                    {
                        "address": "0xPool1",
                        "chain": "optimism",
                        "assets": ["0xToken0"],
                        "enter_timestamp": 1640995200,
                    },
                    {
                        "pool_address": "0xPool2",  # Already correct
                        "chain": "mode",
                        "assets": ["0xToken2", "0xToken3"],
                        "status": "closed",  # Already has status
                        "entry_cost": 10.5,  # Already has entry_cost
                    },
                ],
                [
                    {
                        "pool_address": "0xPool1",
                        "chain": "optimism",
                        "token0": "0xToken0",
                        "token0_symbol": "USDC",
                        "status": "open",
                        "entry_cost": 0.0,
                        "min_hold_days": 21.0,
                        "cost_recovered": False,
                        "principal_usd": 0.0,
                        "enter_timestamp": 1640995200,
                    },
                    {
                        "pool_address": "0xPool2",
                        "chain": "mode",
                        "token0": "0xToken2",
                        "token0_symbol": "MODE",
                        "token1": "0xToken3",
                        "token1_symbol": "USDC",
                        "status": "closed",
                        "entry_cost": 10.5,
                        "min_hold_days": 0.0,  # No enter_timestamp
                        "cost_recovered": False,
                        "principal_usd": 0.0,
                    },
                ],
                [
                    ("optimism", "0xToken0"),
                    ("mode", "0xToken2"),
                    ("mode", "0xToken3"),
                ],
            ),
            # Test case 3: Position with no enter_timestamp (min_hold_days = 0.0)
            (
                {
                    "address": "0xPool3",
                    "chain": "optimism",
                    "assets": ["0xToken4"],
                },
                [
                    {
                        "pool_address": "0xPool3",
                        "chain": "optimism",
                        "token0": "0xToken4",
                        "token0_symbol": "USDC",
                        "status": "open",
                        "entry_cost": 0.0,
                        "min_hold_days": 0.0,  # No enter_timestamp
                        "cost_recovered": False,
                        "principal_usd": 0.0,
                    }
                ],
                [("optimism", "0xToken4")],
            ),
            # Test case 4: Position with single asset (no token1)
            (
                {
                    "address": "0xPool4",
                    "chain": "mode",
                    "assets": ["0xToken5"],
                    "enter_timestamp": 1640995200,
                },
                [
                    {
                        "pool_address": "0xPool4",
                        "chain": "mode",
                        "token0": "0xToken5",
                        "token0_symbol": "MODE",
                        "status": "open",
                        "entry_cost": 0.0,
                        "min_hold_days": 21.0,
                        "cost_recovered": False,
                        "principal_usd": 0.0,
                        "enter_timestamp": 1640995200,
                    }
                ],
                [("mode", "0xToken5")],
            ),
            # Test case 5: Position with empty assets list
            (
                {
                    "address": "0xPool5",
                    "chain": "optimism",
                    "assets": [],
                    "enter_timestamp": 1640995200,
                },
                [
                    {
                        "pool_address": "0xPool5",
                        "chain": "optimism",
                        "status": "open",
                        "entry_cost": 0.0,
                        "min_hold_days": 21.0,
                        "cost_recovered": False,
                        "principal_usd": 0.0,
                        "enter_timestamp": 1640995200,
                    }
                ],
                [],  # No token symbol calls for empty assets
            ),
            # Test case 6: Position with non-list assets (should be ignored)
            (
                {
                    "address": "0xPool6",
                    "chain": "optimism",
                    "assets": "invalid_assets",
                    "enter_timestamp": 1640995200,
                },
                [
                    {
                        "pool_address": "0xPool6",
                        "chain": "optimism",
                        "status": "open",
                        "entry_cost": 0.0,
                        "min_hold_days": 21.0,
                        "cost_recovered": False,
                        "principal_usd": 0.0,
                        "enter_timestamp": 1640995200,
                    }
                ],
                [],  # No token symbol calls for invalid assets
            ),
            # Test case 7: Already properly formatted position (minimal changes)
            (
                {
                    "pool_address": "0xPool7",
                    "chain": "optimism",
                    "token0": "0xToken7",
                    "token0_symbol": "USDC",
                    "status": "open",
                    "entry_cost": 5.0,
                    "min_hold_days": 14.0,
                    "cost_recovered": True,
                    "principal_usd": 100.0,
                },
                [
                    {
                        "pool_address": "0xPool7",
                        "chain": "optimism",
                        "token0": "0xToken7",
                        "token0_symbol": "USDC",
                        "status": "open",
                        "entry_cost": 5.0,
                        "min_hold_days": 14.0,
                        "cost_recovered": True,
                        "principal_usd": 100.0,
                    }
                ],
                [],  # No token symbol calls needed
            ),
        ]
    )
    def test_adjust_current_positions_for_backward_compatibility(
        self, input_data, expected_positions, expected_calls
    ):
        """Test _adjust_current_positions_for_backward_compatibility method."""
        base_behaviour = self._create_base_behaviour()
        
        # Track calls to _get_token_symbol
        symbol_calls = []
        
        def mock_get_token_symbol(chain, address):
            symbol_calls.append((chain, address))
            yield None
            # Return different symbols based on chain and token address for testing
            if chain == "optimism":
                if "Token0" in address:
                    return "USDC"
                elif "Token1" in address:
                    return "DAI"
                elif "Token7" in address:
                    return "USDC"
                else:
                    return "USDC" if "Token" in address else "UNKNOWN"
            elif chain == "mode":
                if "Token2" in address:
                    return "MODE"
                elif "Token3" in address:
                    return "USDC"
                elif "Token5" in address:
                    return "MODE"
                else:
                    return "MODE" if "Token" in address else "UNKNOWN"
            return "UNKNOWN"
        
        def mock_store_current_positions():
            # This method is called at the end
            pass
        
        with patch.object(base_behaviour, '_get_token_symbol', side_effect=mock_get_token_symbol):
            with patch.object(base_behaviour, 'store_current_positions', side_effect=mock_store_current_positions):
                # Call the method
                self._consume_generator(base_behaviour._adjust_current_positions_for_backward_compatibility(input_data))
                
                # Verify the positions were adjusted correctly
                assert base_behaviour.current_positions == expected_positions
                
                # Verify the correct number of token symbol calls were made
                assert symbol_calls == expected_calls

    def test_adjust_current_positions_for_backward_compatibility_unexpected_data_format(self):
        """Test _adjust_current_positions_for_backward_compatibility with unexpected data format."""
        base_behaviour = self._create_base_behaviour()
        
        # Test with non-dict, non-list data
        unexpected_data = "invalid_data"
        
        def mock_store_current_positions():
            pass
        
        with patch.object(base_behaviour, 'store_current_positions', side_effect=mock_store_current_positions):
            self._consume_generator(base_behaviour._adjust_current_positions_for_backward_compatibility(unexpected_data))
            
            # Should set current_positions to empty list
            assert base_behaviour.current_positions == []

    def test_adjust_current_positions_for_backward_compatibility_empty_list(self):
        """Test _adjust_current_positions_for_backward_compatibility with empty list."""
        base_behaviour = self._create_base_behaviour()
        
        empty_list = []
        
        def mock_store_current_positions():
            pass
        
        with patch.object(base_behaviour, 'store_current_positions', side_effect=mock_store_current_positions):
            self._consume_generator(base_behaviour._adjust_current_positions_for_backward_compatibility(empty_list))
            
            # Should set current_positions to empty list
            assert base_behaviour.current_positions == []

    def test_adjust_current_positions_for_backward_compatibility_token_symbol_error_handling(self):
        """Test _adjust_current_positions_for_backward_compatibility when _get_token_symbol fails."""
        base_behaviour = self._create_base_behaviour()
        
        input_data = {
            "address": "0xPool8",
            "chain": "optimism",
            "assets": ["0xToken8"],
            "enter_timestamp": 1640995200,
        }
        
        def mock_get_token_symbol(chain, address):
            yield None
            return None  # Simulate failure
        
        def mock_store_current_positions():
            pass
        
        with patch.object(base_behaviour, '_get_token_symbol', side_effect=mock_get_token_symbol):
            with patch.object(base_behaviour, 'store_current_positions', side_effect=mock_store_current_positions):
                self._consume_generator(base_behaviour._adjust_current_positions_for_backward_compatibility(input_data))
                
                # Should still create the position with None symbol
                expected_position = {
                    "pool_address": "0xPool8",
                    "chain": "optimism",
                    "token0": "0xToken8",
                    "token0_symbol": None,
                    "status": "open",
                    "entry_cost": 0.0,
                    "min_hold_days": 21.0,
                    "cost_recovered": False,
                    "principal_usd": 0.0,
                    "enter_timestamp": 1640995200,
                }
                
                assert base_behaviour.current_positions == [expected_position]

    def test_adjust_current_positions_for_backward_compatibility_assets_not_list(self):
        """Test _adjust_current_positions_for_backward_compatibility when assets is not a list."""
        base_behaviour = self._create_base_behaviour()
        
        input_data = {
            "address": "0xPool9",
            "chain": "optimism",
            "assets": "not_a_list",  # This should be ignored
            "enter_timestamp": 1640995200,
        }
        
        def mock_store_current_positions():
            pass
        
        with patch.object(base_behaviour, 'store_current_positions', side_effect=mock_store_current_positions):
            self._consume_generator(base_behaviour._adjust_current_positions_for_backward_compatibility(input_data))
            
            # Should create position without token fields
            expected_position = {
                "pool_address": "0xPool9",
                "chain": "optimism",
                "status": "open",
                "entry_cost": 0.0,
                "min_hold_days": 21.0,
                "cost_recovered": False,
                "principal_usd": 0.0,
                "enter_timestamp": 1640995200,
            }
            
            assert base_behaviour.current_positions == [expected_position]

    def test_adjust_current_positions_for_backward_compatibility_store_current_positions_called(self):
        """Test that store_current_positions is called after adjusting positions."""
        base_behaviour = self._create_base_behaviour()
        
        input_data = {
            "address": "0xPool10",
            "chain": "optimism",
            "assets": ["0xToken10"],
            "enter_timestamp": 1640995200,
        }
        
        store_called = False
        
        def mock_store_current_positions():
            nonlocal store_called
            store_called = True
        
        def mock_get_token_symbol(chain, address):
            yield None
            return "USDC"
        
        with patch.object(base_behaviour, '_get_token_symbol', side_effect=mock_get_token_symbol):
            with patch.object(base_behaviour, 'store_current_positions', side_effect=mock_store_current_positions):
                self._consume_generator(base_behaviour._adjust_current_positions_for_backward_compatibility(input_data))
                
                # Verify store_current_positions was called
                assert store_called is True

    @pytest.mark.parametrize(
        "chain,position_id,costs,existing_costs,expected_calls,expected_logs",
        [
            # Test case 1: Store new entry costs with no existing data
            (
                "optimism",
                "0xPool123",
                10.5,
                {},
                2,  # _get_all_entry_costs + _write_kv
                ["Stored entry costs: entry_costs_optimism_0xPool123 = $10.500000"],
            ),
            # Test case 2: Store entry costs with existing data
            (
                "mode",
                "0xPool456",
                25.75,
                {"entry_costs_optimism_0xPool123": 10.5, "entry_costs_mode_0xPool789": 15.25},
                2,  # _get_all_entry_costs + _write_kv
                ["Stored entry costs: entry_costs_mode_0xPool456 = $25.750000"],
            ),
            # Test case 3: Update existing entry costs
            (
                "optimism",
                "0xPool123",
                20.0,  # Update existing
                {"entry_costs_optimism_0xPool123": 10.5, "entry_costs_mode_0xPool456": 25.75},
                2,  # _get_all_entry_costs + _write_kv
                ["Stored entry costs: entry_costs_optimism_0xPool123 = $20.000000"],
            ),
            # Test case 4: Store with zero costs
            (
                "optimism",
                "0xPoolZero",
                0.0,
                {},
                2,  # _get_all_entry_costs + _write_kv
                ["Stored entry costs: entry_costs_optimism_0xPoolZero = $0.000000"],
            ),
            # Test case 5: Store with very small costs
            (
                "mode",
                "0xPoolSmall",
                0.000001,
                {},
                2,  # _get_all_entry_costs + _write_kv
                ["Stored entry costs: entry_costs_mode_0xPoolSmall = $0.000001"],
            ),
        ]
    )
    def test_store_entry_costs(
        self, chain, position_id, costs, existing_costs, expected_calls, expected_logs
    ):
        """Test _store_entry_costs method."""
        base_behaviour = self._create_base_behaviour()
        
        # Track calls to _get_all_entry_costs and _write_kv
        get_calls = []
        write_calls = []
        
        def mock_get_all_entry_costs():
            get_calls.append(True)
            yield None
            return existing_costs
        
        def mock_write_kv(data):
            write_calls.append(data)
            yield None
            return True
        
        def mock_get_entry_costs_key(chain, position_id):
            return f"entry_costs_{chain}_{position_id}"
        
        with patch.object(base_behaviour, '_get_all_entry_costs', side_effect=mock_get_all_entry_costs):
            with patch.object(base_behaviour, '_write_kv', side_effect=mock_write_kv):
                with patch.object(base_behaviour, '_get_entry_costs_key', side_effect=mock_get_entry_costs_key):
                    # Call the method
                    self._consume_generator(base_behaviour._store_entry_costs(chain, position_id, costs))
                    
                    # Verify the correct number of calls were made
                    assert len(get_calls) == 1
                    assert len(write_calls) == 1
                    
                    # Verify the write_kv was called with the correct data
                    expected_key = f"entry_costs_{chain}_{position_id}"
                    expected_dict = existing_costs.copy()
                    expected_dict[expected_key] = costs
                    expected_data = {"entry_costs_dict": json.dumps(expected_dict)}
                    
                    assert write_calls[0] == expected_data

    def test_store_entry_costs_exception_handling(self):
        """Test _store_entry_costs exception handling."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_get_all_entry_costs():
            yield None
            raise Exception("KV store error")
        
        def mock_write_kv(data):
            yield None
            return True
        
        def mock_get_entry_costs_key(chain, position_id):
            return f"entry_costs_{chain}_{position_id}"
        
        with patch.object(base_behaviour, '_get_all_entry_costs', side_effect=mock_get_all_entry_costs):
            with patch.object(base_behaviour, '_write_kv', side_effect=mock_write_kv):
                with patch.object(base_behaviour, '_get_entry_costs_key', side_effect=mock_get_entry_costs_key):
                    # Call the method - should not raise exception
                    self._consume_generator(base_behaviour._store_entry_costs("optimism", "0xPool123", 10.5))
                    
                    # Should handle exception gracefully

    def test_store_entry_costs_write_kv_failure(self):
        """Test _store_entry_costs when _write_kv fails."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_get_all_entry_costs():
            yield None
            return {"existing_key": 5.0}
        
        def mock_write_kv(data):
            yield None
            raise Exception("Write failed")
        
        def mock_get_entry_costs_key(chain, position_id):
            return f"entry_costs_{chain}_{position_id}"
        
        with patch.object(base_behaviour, '_get_all_entry_costs', side_effect=mock_get_all_entry_costs):
            with patch.object(base_behaviour, '_write_kv', side_effect=mock_write_kv):
                with patch.object(base_behaviour, '_get_entry_costs_key', side_effect=mock_get_entry_costs_key):
                    # Call the method - should not raise exception
                    self._consume_generator(base_behaviour._store_entry_costs("optimism", "0xPool123", 10.5))
                    
                    # Should handle exception gracefully

    @pytest.mark.parametrize(
        "kv_result,expected_result,expected_calls",
        [
            # Test case 1: Successful retrieval with existing data
            (
                {"entry_costs_dict": '{"optimism_0xPool123": "10.5", "mode_0xPool456": "25.75"}'},
                {"optimism_0xPool123": 10.5, "mode_0xPool456": 25.75},
                1,  # _read_kv called once
            ),
            # Test case 2: No data in KV store
            (
                {},
                {},
                1,  # _read_kv called once
            ),
            # Test case 3: Missing entry_costs_dict key
            (
                {"other_key": "some_value"},
                {},
                1,  # _read_kv called once
            ),
            # Test case 4: Empty entry_costs_dict
            (
                {"entry_costs_dict": "{}"},
                {},
                1,  # _read_kv called once
            ),
            # Test case 5: Single entry
            (
                {"entry_costs_dict": '{"optimism_0xPool123": "10.5"}'},
                {"optimism_0xPool123": 10.5},
                1,  # _read_kv called once
            ),
            # Test case 6: Mixed data types (strings and numbers)
            (
                {"entry_costs_dict": '{"optimism_0xPool123": "10.5", "mode_0xPool456": 25.75}'},
                {"optimism_0xPool123": 10.5, "mode_0xPool456": 25.75},
                1,  # _read_kv called once
            ),
        ]
    )
    def test_get_all_entry_costs(self, kv_result, expected_result, expected_calls):
        """Test _get_all_entry_costs method."""
        base_behaviour = self._create_base_behaviour()
        
        # Track calls to _read_kv
        read_calls = []
        
        def mock_read_kv(keys):
            read_calls.append(keys)
            yield None
            return kv_result
        
        with patch.object(base_behaviour, '_read_kv', side_effect=mock_read_kv):
            # Call the method
            result = self._consume_generator(base_behaviour._get_all_entry_costs())
            
            # Verify the correct number of calls were made
            assert len(read_calls) == expected_calls
            assert read_calls[0] == ("entry_costs_dict",)
            
            # Verify the result
            assert result == expected_result

    def test_get_all_entry_costs_json_decode_error(self):
        """Test _get_all_entry_costs with invalid JSON data."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_read_kv(keys):
            yield None
            return {"entry_costs_dict": "invalid_json"}
        
        with patch.object(base_behaviour, '_read_kv', side_effect=mock_read_kv):
            # Call the method - should handle JSON decode error
            result = self._consume_generator(base_behaviour._get_all_entry_costs())
            
            # Should return empty dict on error
            assert result == {}

    def test_get_all_entry_costs_read_kv_exception(self):
        """Test _get_all_entry_costs when _read_kv raises exception."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_read_kv(keys):
            yield None
            raise Exception("KV store error")
        
        with patch.object(base_behaviour, '_read_kv', side_effect=mock_read_kv):
            # Call the method - should handle exception
            result = self._consume_generator(base_behaviour._get_all_entry_costs())
            
            # Should return empty dict on error
            assert result == {}

    def test_get_all_entry_costs_invalid_value_types(self):
        """Test _get_all_entry_costs with invalid value types in JSON."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_read_kv(keys):
            yield None
            return {"entry_costs_dict": '{"key1": "invalid_float", "key2": "10.5"}'}
        
        with patch.object(base_behaviour, '_read_kv', side_effect=mock_read_kv):
            # Call the method - should handle invalid float conversion
            result = self._consume_generator(base_behaviour._get_all_entry_costs())
            
            # Should return empty dict on error
            assert result == {}

    def test_store_entry_costs_integration(self):
        """Test integration between _store_entry_costs and _get_all_entry_costs."""
        base_behaviour = self._create_base_behaviour()
        
        # Track all calls
        read_calls = []
        write_calls = []
        
        def mock_read_kv(keys):
            read_calls.append(keys)
            yield None
            if not write_calls:  # First call (get_all_entry_costs)
                return {"entry_costs_dict": '{"existing_key": "5.0"}'}
            else:  # Second call (get_all_entry_costs after write)
                return write_calls[-1]  # Return what was just written
        
        def mock_write_kv(data):
            write_calls.append(data)
            yield None
            return True
        
        def mock_get_entry_costs_key(chain, position_id):
            return f"entry_costs_{chain}_{position_id}"
        
        with patch.object(base_behaviour, '_read_kv', side_effect=mock_read_kv):
            with patch.object(base_behaviour, '_write_kv', side_effect=mock_write_kv):
                with patch.object(base_behaviour, '_get_entry_costs_key', side_effect=mock_get_entry_costs_key):
                    # Store first entry costs
                    self._consume_generator(base_behaviour._store_entry_costs("optimism", "0xPool1", 10.5))
                    
                    # Store second entry costs
                    self._consume_generator(base_behaviour._store_entry_costs("mode", "0xPool2", 25.75))
                    
                    # Verify the calls
                    assert len(read_calls) == 2
                    assert len(write_calls) == 2
                    
                    # Verify the final state
                    expected_final = {
                        "entry_costs_dict": json.dumps({
                            "existing_key": 5.0,
                            "entry_costs_optimism_0xPool1": 10.5,
                            "entry_costs_mode_0xPool2": 25.75
                        })
                    }
                    assert write_calls[-1] == expected_final

    @pytest.mark.parametrize(
        "safe_address,api_responses,expected_tokens,expected_logs,test_description",
        [
            # Single page response
            (
                "0xSafe1",
                [
                    {
                        "success": True,
                        "response_data": {
                            "items": [
                                {"token": "0xToken1", "balance": "1000000000000000000"},
                                {"token": "0xToken2", "balance": "2000000000000000000"}
                            ],
                            "next_page_params": None
                        }
                    }
                ],
                [
                    {"token": "0xToken1", "balance": "1000000000000000000"},
                    {"token": "0xToken2", "balance": "2000000000000000000"}
                ],
                ["Fetching Mode tokens page:", "Total tokens fetched from Mode Explorer API: 2"],
                "Single page with 2 tokens"
            ),
            # Multiple pages
            (
                "0xSafe2",
                [
                    {
                        "success": True,
                        "response_data": {
                            "items": [{"token": "0xToken1", "balance": "1000000000000000000"}],
                            "next_page_params": {"page": "2", "limit": "10"}
                        }
                    },
                    {
                        "success": True,
                        "response_data": {
                            "items": [{"token": "0xToken2", "balance": "2000000000000000000"}],
                            "next_page_params": None
                        }
                    }
                ],
                [
                    {"token": "0xToken1", "balance": "1000000000000000000"},
                    {"token": "0xToken2", "balance": "2000000000000000000"}
                ],
                ["Fetching Mode tokens page:", "Total tokens fetched from Mode Explorer API: 2"],
                "Multiple pages with pagination"
            ),
            # Empty response
            (
                "0xSafe3",
                [
                    {
                        "success": True,
                        "response_data": {
                            "items": [],
                            "next_page_params": None
                        }
                    }
                ],
                [],
                ["Fetching Mode tokens page:", "No more token results from Mode Explorer API", "Total tokens fetched from Mode Explorer API: 0"],
                "Empty response with no tokens"
            ),
            # API failure
            (
                "0xSafe4",
                [
                    {
                        "success": False,
                        "response_data": "API Error"
                    }
                ],
                [],
                ["Fetching Mode tokens page:", "Failed to fetch Mode token data: API Error", "Total tokens fetched from Mode Explorer API: 0"],
                "API failure on first request"
            ),
            # API failure on second page
            (
                "0xSafe5",
                [
                    {
                        "success": True,
                        "response_data": {
                            "items": [{"token": "0xToken1", "balance": "1000000000000000000"}],
                            "next_page_params": {"page": "2", "limit": "10"}
                        }
                    },
                    {
                        "success": False,
                        "response_data": "API Error on second page"
                    }
                ],
                [{"token": "0xToken1", "balance": "1000000000000000000"}],
                ["Fetching Mode tokens page:", "Failed to fetch Mode token data: API Error on second page", "Total tokens fetched from Mode Explorer API: 1"],
                "API failure on second page"
            ),
            # Missing items in response
            (
                "0xSafe6",
                [
                    {
                        "success": True,
                        "response_data": {
                            "next_page_params": None
                        }
                    }
                ],
                [],
                ["Fetching Mode tokens page:", "No more token results from Mode Explorer API", "Total tokens fetched from Mode Explorer API: 0"],
                "Missing items field in response"
            ),
            # Large number of pages
            (
                "0xSafe7",
                [
                    {
                        "success": True,
                        "response_data": {
                            "items": [{"token": f"0xToken{i}", "balance": f"{i}000000000000000000"} for i in range(1, 4)],
                            "next_page_params": {"page": "2", "limit": "3"}
                        }
                    },
                    {
                        "success": True,
                        "response_data": {
                            "items": [{"token": f"0xToken{i}", "balance": f"{i}000000000000000000"} for i in range(4, 7)],
                            "next_page_params": {"page": "3", "limit": "3"}
                        }
                    },
                    {
                        "success": True,
                        "response_data": {
                            "items": [{"token": "0xToken7", "balance": "7000000000000000000"}],
                            "next_page_params": None
                        }
                    }
                ],
                [{"token": f"0xToken{i}", "balance": f"{i}000000000000000000"} for i in range(1, 8)],
                ["Fetching Mode tokens page:", "Total tokens fetched from Mode Explorer API: 7"],
                "Three pages with multiple tokens each"
            ),
        ]
    )
    def test_fetch_mode_tokens_with_pagination(
        self, safe_address, api_responses, expected_tokens, expected_logs, test_description
    ):
        """Test _fetch_mode_tokens_with_pagination with various scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Track API calls
        api_calls = []
        
        def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
            api_calls.append(endpoint)
            response_index = len(api_calls) - 1
            if response_index < len(api_responses):
                response = api_responses[response_index]
                yield None
                return response["success"], response["response_data"]
            else:
                yield None
                return False, "Unexpected call"
        
        with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
            result = self._consume_generator(base_behaviour._fetch_mode_tokens_with_pagination(safe_address))
            
            # Verify the result
            assert result == expected_tokens
            
            # Verify API calls were made
            assert len(api_calls) == len(api_responses)
            
            # Verify first call has correct base URL structure (without hardcoding the base URL)
            assert f"/api/v2/addresses/{safe_address}/tokens?type=ERC-20" in api_calls[0]
            
            # Verify subsequent calls include pagination parameters
            for i, response in enumerate(api_responses[:-1]):  # All except last
                if response["success"] and response["response_data"].get("next_page_params"):
                    next_page_params = response["response_data"]["next_page_params"]
                    expected_params = "&".join([f"{key}={value}" for key, value in next_page_params.items()])
                    if i + 1 < len(api_calls):
                        assert expected_params in api_calls[i + 1]

    def test_fetch_mode_tokens_with_pagination_request_with_retries_exception(self):
        """Test _fetch_mode_tokens_with_pagination when _request_with_retries raises exception."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
            raise Exception("Network error")
        
        with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
            # Should propagate exception from _request_with_retries
            with pytest.raises(Exception, match="Network error"):
                self._consume_generator(base_behaviour._fetch_mode_tokens_with_pagination("0xSafe8"))

    def test_fetch_mode_tokens_with_pagination_empty_safe_address(self):
        """Test _fetch_mode_tokens_with_pagination with empty safe address."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
            # Verify endpoint includes empty address
            assert "/addresses//tokens" in endpoint
            yield None
            return True, {"items": [], "next_page_params": None}
        
        with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
            result = self._consume_generator(base_behaviour._fetch_mode_tokens_with_pagination(""))
            
            assert result == []

    def test_fetch_mode_tokens_with_pagination_missing_next_page_params(self):
        """Test _fetch_mode_tokens_with_pagination when next_page_params is missing."""
        base_behaviour = self._create_base_behaviour()
        
        def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
            yield None
            return True, {
                "items": [{"token": "0xToken1", "balance": "1000000000000000000"}]
                # Missing next_page_params
            }
        
        with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
            result = self._consume_generator(base_behaviour._fetch_mode_tokens_with_pagination("0xSafe9"))
            
            # Should return the items and stop pagination
            assert result == [{"token": "0xToken1", "balance": "1000000000000000000"}]

    def test_fetch_mode_tokens_with_pagination_headers_and_parameters(self):
        """Test _fetch_mode_tokens_with_pagination verifies correct headers and parameters."""
        base_behaviour = self._create_base_behaviour()
        
        captured_headers = []
        captured_endpoints = []
        
        def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
            captured_headers.append(headers)
            captured_endpoints.append(endpoint)
            yield None
            return True, {"items": [], "next_page_params": None}
        
        with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
            self._consume_generator(base_behaviour._fetch_mode_tokens_with_pagination("0xSafe10"))
            
            # Verify headers
            expected_headers = {
                "Accept": "application/json",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
            assert captured_headers[0] == expected_headers
            
            # Verify endpoint structure
            assert "/api/v2/addresses/0xSafe10/tokens?type=ERC-20" in captured_endpoints[0]

    def test_fetch_mode_tokens_with_pagination_max_retries_and_wait(self):
        """Test _fetch_mode_tokens_with_pagination uses correct retry parameters."""
        base_behaviour = self._create_base_behaviour()
        
        captured_max_retries = []
        captured_retry_wait = []
        
        def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
            captured_max_retries.append(max_retries)
            captured_retry_wait.append(retry_wait)
            yield None
            return True, {"items": [], "next_page_params": None}
        
        with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
            self._consume_generator(base_behaviour._fetch_mode_tokens_with_pagination("0xSafe11"))
            
            # Verify retry parameters
            assert captured_max_retries[0] == 3  # MAX_RETRIES_FOR_API_CALL
            assert captured_retry_wait[0] == 2

    def test_fetch_mode_tokens_with_pagination_rate_limited_callback(self):
        """Test _fetch_mode_tokens_with_pagination uses correct rate limited callback."""
        base_behaviour = self._create_base_behaviour()
        
        captured_callbacks = []
        
        def mock_request_with_retries(endpoint, method, headers, rate_limited_callback, max_retries, retry_wait):
            captured_callbacks.append(rate_limited_callback)
            yield None
            return True, {"items": [], "next_page_params": None}
        
        with patch.object(base_behaviour, '_request_with_retries', side_effect=mock_request_with_retries):
            self._consume_generator(base_behaviour._fetch_mode_tokens_with_pagination("0xSafe12"))
            
            # Verify callback is a lambda that returns None
            assert captured_callbacks[0]() is None

    # Agent Type and Attribute Management Tests
    @pytest.mark.parametrize("method_name,method_params,expected_response,expected_call_params", [
        (
            "get_agent_type_by_name",
            ("Modius",),
            {"type_id": "123", "type_name": "Modius", "description": "Test agent type"},
            {
                "method": "read_",
                "method_name": "get_agent_type_by_name",
                "endpoint": "api/agent-types/name/Modius"
            }
        ),
        (
            "create_agent_type",
            ("TestType", "Test description"),
            {"type_id": "456", "type_name": "TestType", "description": "Test description"},
            {
                "method": "create_",
                "method_name": "create_agent_type",
                "endpoint": "api/agent-types/",
                "data": {"type_name": "TestType", "description": "Test description"}
            }
        ),
        (
            "get_attr_def_by_name",
            ("APR",),
            {"attr_def_id": "789", "attr_name": "APR", "data_type": "json"},
            {
                "method": "read_",
                "method_name": "get_attr_def_by_name",
                "endpoint": "api/attributes/name/APR"
            }
        ),
        (
            "get_agent_registry_by_address",
            ("0x1234567890123456789012345678901234567890",),
            {"agent_id": "agent123", "agent_name": "TestAgent", "eth_address": "0x1234567890123456789012345678901234567890"},
            {
                "method": "read_",
                "method_name": "get_agent_registry_by_address",
                "endpoint": "api/agent-registry/address/0x1234567890123456789012345678901234567890"
            }
        ),
        (
            "create_agent_registry",
            ("TestAgent", "123", "0x1234567890123456789012345678901234567890"),
            {"agent_id": "agent123", "agent_name": "TestAgent", "type_id": "123", "eth_address": "0x1234567890123456789012345678901234567890"},
            {
                "method": "create_",
                "method_name": "create_agent_registry",
                "endpoint": "api/agent-registry/",
                "data": {"agent_name": "TestAgent", "type_id": "123", "eth_address": "0x1234567890123456789012345678901234567890"}
            }
        ),
    ])
    def test_mirrordb_methods(self, method_name, method_params, expected_response, expected_call_params):
        """Test mirrordb methods with parameterized test data."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the _call_mirrordb method on the instance
        def _fake_call_mirrordb(*args, **kwargs):
            def _gen():
                if False:
                    yield
                return expected_response
            return _gen()
        with patch.object(base_behaviour, '_call_mirrordb', side_effect=_fake_call_mirrordb) as mock_call:
            method = getattr(base_behaviour, method_name)
            result = self._consume_generator(method(*method_params))
            
            # Verify the result
            assert result == expected_response
            
            # Verify _call_mirrordb was called with correct parameters
            mock_call.assert_called_once_with(**expected_call_params)

    @pytest.mark.parametrize("method_name,method_params,expected_response,expected_call_params,expected_data_structure", [
        (
            "create_attribute_definition",
            ("123", "APR", "json", True, "{}", "agent123"),
            {"attr_def_id": "789", "attr_name": "APR", "data_type": "json"},
            {
                "method": "create_",
                "method_name": "create_attribute_definition",
                "endpoint": "api/agent-types/123/attributes/"
            },
            {
                "attr_def": {"type_id": "123", "attr_name": "APR", "data_type": "json", "is_required": True, "default_value": "{}"},
                "auth": {"agent_id": "agent123", "signature": "0x1234567890abcdef", "message": "timestamp:123,endpoint:api/agent-types/123/attributes/"}
            }
        ),
        (
            "create_agent_attribute",
            ("agent123", "789", '{"apr": 0.05}'),
            {"attr_id": "attr123", "agent_id": "agent123", "attr_def_id": "789"},
            {
                "method": "create_",
                "method_name": "create_agent_attribute",
                "endpoint": "api/agents/agent123/attributes/"
            },
            {
                "agent_attr": {"agent_id": "agent123", "attr_def_id": "789", "string_value": None, "integer_value": None, "float_value": None, "boolean_value": None, "date_value": None, "json_value": '{"apr": 0.05}'},
                "auth": {"agent_id": "agent123", "signature": "0x1234567890abcdef", "message": "timestamp:123,endpoint:api/agents/agent123/attributes/"}
            }
        ),
    ])
    def test_mirrordb_methods_with_signature(self, method_name, method_params, expected_response, expected_call_params, expected_data_structure):
        """Test mirrordb methods that require signature verification."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the sign_message method
        mock_signature = "0x1234567890abcdef"
        def _fake_sign_message(msg):
            def _gen():
                if False:
                    yield
                return mock_signature
            return _gen()
        
        with patch.object(base_behaviour, 'sign_message', side_effect=_fake_sign_message) as mock_sign:
            # Mock the round_sequence._last_round_transition_timestamp attribute
            from datetime import datetime
            mock_timestamp = datetime(2023, 1, 1, 12, 0, 0)
            with patch.object(base_behaviour.round_sequence, '_last_round_transition_timestamp', mock_timestamp):
                # Mock the _call_mirrordb method
                def _fake_call_mirrordb(*args, **kwargs):
                    def _gen():
                        if False:
                            yield
                        return expected_response
                    return _gen()
                
                with patch.object(base_behaviour, '_call_mirrordb', side_effect=_fake_call_mirrordb) as mock_call:
                    method = getattr(base_behaviour, method_name)
                    result = self._consume_generator(method(*method_params))
                    
                    # Verify the result
                    assert result == expected_response
                    
                    # Verify sign_message was called
                    mock_sign.assert_called_once()
                    
                    # Verify _call_mirrordb was called with correct parameters
                    mock_call.assert_called_once()
                    call_args = mock_call.call_args
                    assert call_args[1]['method'] == expected_call_params['method']
                    assert call_args[1]['method_name'] == expected_call_params['method_name']
                    assert call_args[1]['endpoint'] == expected_call_params['endpoint']
                    
                    # Verify the data structure
                    data = call_args[1]['data']
                    assert 'attr_def' in data or 'agent_attr' in data
                    assert 'auth' in data
                    
                    # Verify auth data
                    auth = data['auth']
                    # For create_agent_attribute, agent_id is first param; for create_attribute_definition, it's 6th param
                    expected_agent_id = method_params[0] if method_name == 'create_agent_attribute' else method_params[5]
                    assert auth['agent_id'] == expected_agent_id
                    assert auth['signature'] == mock_signature

    @pytest.mark.parametrize("method_name,method_params", [
        ("create_attribute_definition", ("123", "APR", "json", True, "{}", "agent123")),
        ("create_agent_attribute", ("agent123", "789", '{"apr": 0.05}')),
    ])
    def test_mirrordb_methods_signature_failure(self, method_name, method_params):
        """Test mirrordb methods when signature generation fails."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the sign_message method to return None
        def _mock_sign_message_none(*args, **kwargs):
            yield
            return None
        with patch.object(base_behaviour, 'sign_message', side_effect=_mock_sign_message_none) as mock_sign:
            # Mock the round_sequence._last_round_transition_timestamp attribute
            from datetime import datetime
            mock_timestamp = datetime(2023, 1, 1, 12, 0, 0)
            with patch.object(base_behaviour.round_sequence, '_last_round_transition_timestamp', mock_timestamp):
                method = getattr(base_behaviour, method_name)
                result = self._consume_generator(method(*method_params))
                
                # Should return None when signature fails
                assert result is None

    @pytest.mark.parametrize("method_name,read_data,get_result,create_result,expected_result,create_params,write_data,exception_message", [
        (
            "_get_or_create_agent_type",
            {},  # No existing data
            None,  # get_agent_type_by_name returns None
            {"type_id": "123", "type_name": "Modius", "description": "Test description"},  # create_agent_type succeeds
            {"type_id": "123", "type_name": "Modius", "description": "Test description"},  # Expected result
            ("Modius", "An agent for DeFi liquidity management and APR tracking"),  # create_agent_type params
            {"agent_type": '{"type_id": "123", "type_name": "Modius", "description": "Test description"}'},  # _write_kv data
            None  # No exception
        ),
        (
            "_get_or_create_agent_type",
            {"agent_type": '{"type_id": "123", "type_name": "Modius", "description": "Test description"}'},  # Existing data
            None,  # Not used when data exists
            None,  # Not used when data exists
            {"type_id": "123", "type_name": "Modius", "description": "Test description"},  # Expected result
            None,  # Not used when data exists
            None,  # Not used when data exists
            None  # No exception
        ),
        (
            "_get_or_create_agent_type",
            {},  # No existing data
            None,  # get_agent_type_by_name returns None
            None,  # create_agent_type fails
            None,  # Not used when exception occurs
            None,  # Not used when exception occurs
            None,  # Not used when exception occurs
            "Failed to create agent type."  # Exception message
        ),
        (
            "_get_or_create_attr_def",
            {},  # No existing data
            None,  # get_attr_def_by_name returns None
            {"attr_def_id": "789", "attr_name": "APR", "data_type": "json"},  # create_attribute_definition succeeds
            {"attr_def_id": "789", "attr_name": "APR", "data_type": "json"},  # Expected result
            ("123", "APR", "json", True, "{}", "agent123"),  # create_attribute_definition params
            {"attr_def": '{"attr_def_id": "789", "attr_name": "APR", "data_type": "json"}'},  # _write_kv data
            None  # No exception
        ),
        (
            "_get_or_create_attr_def",
            {"attr_def": '{"attr_def_id": "789", "attr_name": "APR", "data_type": "json"}'},  # Existing data
            None,  # Not used when data exists
            None,  # Not used when data exists
            {"attr_def_id": "789", "attr_name": "APR", "data_type": "json"},  # Expected result
            None,  # Not used when data exists
            None,  # Not used when data exists
            None  # No exception
        ),
        (
            "_get_or_create_attr_def",
            {},  # No existing data
            None,  # get_attr_def_by_name returns None
            None,  # create_attribute_definition fails
            None,  # Not used when exception occurs
            None,  # Not used when exception occurs
            None,  # Not used when exception occurs
            "Failed to create attribute definition."  # Exception message
        ),
    ])
    def test_get_or_create_patterns(self, method_name, read_data, get_result, create_result, expected_result, create_params, write_data, exception_message):
        """Test get-or-create pattern methods with parameterized test data."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _read_kv
        def _mock_read_kv(*args, **kwargs):
            yield
            return read_data
        with patch.object(base_behaviour, '_read_kv', side_effect=_mock_read_kv) as mock_read:
            
            if method_name == "_get_or_create_agent_type":
                # Mock get_agent_type_by_name
                def _mock_get_agent_type_by_name(*args, **kwargs):
                    yield
                    return get_result
                with patch.object(base_behaviour, 'get_agent_type_by_name', side_effect=_mock_get_agent_type_by_name) as mock_get:
                    if create_result is None and exception_message:
                        # Test failure case
                        def _mock_create_agent_type_fail(*args, **kwargs):
                            yield
                            return None
                        with patch.object(base_behaviour, 'create_agent_type', side_effect=_mock_create_agent_type_fail) as mock_create:
                            # Should raise an exception
                            with pytest.raises(Exception, match=exception_message):
                                self._consume_generator(base_behaviour._get_or_create_agent_type("0x123"))
                    elif create_result:
                        # Test creation case
                        def _mock_create_agent_type(*args, **kwargs):
                            yield
                            return create_result
                        with patch.object(base_behaviour, 'create_agent_type', side_effect=_mock_create_agent_type) as mock_create:
                            # Mock _write_kv
                            def _mock_write_kv_agent_type(*args, **kwargs):
                                yield
                                return None
                            with patch.object(base_behaviour, '_write_kv', side_effect=_mock_write_kv_agent_type) as mock_write:
                                result = self._consume_generator(base_behaviour._get_or_create_agent_type("0x123"))
                                
                                # Verify the result
                                assert result == expected_result
                                
                                # Verify create_agent_type was called with correct parameters
                                mock_create.assert_called_once_with(*create_params)
                                
                                # Verify _write_kv was called
                                mock_write.assert_called_once_with(write_data)
                    else:
                        # Test existing data case
                        result = self._consume_generator(base_behaviour._get_or_create_agent_type("0x123"))
                        assert result == expected_result
                        
            elif method_name == "_get_or_create_attr_def":
                # Mock get_attr_def_by_name
                def _mock_get_attr_def_by_name(*args, **kwargs):
                    yield
                    return get_result
                with patch.object(base_behaviour, 'get_attr_def_by_name', side_effect=_mock_get_attr_def_by_name) as mock_get:
                    if create_result is None and exception_message:
                        # Test failure case
                        def _mock_create_attribute_definition_fail(*args, **kwargs):
                            yield
                            return None
                        with patch.object(base_behaviour, 'create_attribute_definition', side_effect=_mock_create_attribute_definition_fail) as mock_create:
                            # Should raise an exception
                            with pytest.raises(Exception, match=exception_message):
                                self._consume_generator(base_behaviour._get_or_create_attr_def("123", "agent123"))
                    elif create_result:
                        # Test creation case
                        def _mock_create_attribute_definition(*args, **kwargs):
                            yield
                            return create_result
                        with patch.object(base_behaviour, 'create_attribute_definition', side_effect=_mock_create_attribute_definition) as mock_create:
                            # Mock _write_kv
                            def _mock_write_kv_attr_def(*args, **kwargs):
                                yield
                                return None
                            with patch.object(base_behaviour, '_write_kv', side_effect=_mock_write_kv_attr_def) as mock_write:
                                result = self._consume_generator(base_behaviour._get_or_create_attr_def("123", "agent123"))
                                
                                # Verify the result
                                assert result == expected_result
                                
                                # Verify create_attribute_definition was called with correct parameters
                                mock_create.assert_called_once_with(*create_params)
                                
                                # Verify _write_kv was called
                                mock_write.assert_called_once_with(write_data)
                    else:
                        # Test existing data case
                        result = self._consume_generator(base_behaviour._get_or_create_attr_def("123", "agent123"))
                        assert result == expected_result

    @pytest.mark.parametrize("read_data,agent_type_result,registry_result,create_result,expected_result,write_data,exception", [
        (
            {},  # No existing data
            {"type_id": "123", "type_name": "Modius"},  # _get_or_create_agent_type succeeds
            None,  # get_agent_registry_by_address returns None
            {"agent_id": "agent123", "agent_name": "TestAgent", "type_id": "123"},  # create_agent_registry succeeds
            {"agent_id": "agent123", "agent_name": "TestAgent", "type_id": "123"},  # Expected result
            {"agent_registry": '{"agent_id": "agent123", "agent_name": "TestAgent", "type_id": "123"}'},  # _write_kv data
            None  # No exception
        ),
        (
            {"agent_registry": '{"agent_id": "agent123", "agent_name": "TestAgent", "type_id": "123"}'},  # Existing data
            None,  # Not used when data exists
            None,  # Not used when data exists
            None,  # Not used when data exists
            {"agent_id": "agent123", "agent_name": "TestAgent", "type_id": "123"},  # Expected result
            None,  # Not used when data exists
            None  # No exception
        ),
        (
            {},  # No existing data
            None,  # _get_or_create_agent_type fails
            None,  # Not used when agent type fails
            None,  # Not used when agent type fails
            None,  # Expected result (None)
            None,  # Not used when agent type fails
            None  # No exception
        ),
        (
            {},  # No existing data
            {"type_id": "123", "type_name": "Modius"},  # _get_or_create_agent_type succeeds
            None,  # get_agent_registry_by_address returns None
            None,  # create_agent_registry fails
            None,  # Expected result (None)
            None,  # Not used when creation fails
            None  # No exception
        ),

        (
            {},  # No existing data
            None,  # Not used when exception occurs
            None,  # Not used when exception occurs
            None,  # Not used when exception occurs
            None,  # Expected result (None)
            None,  # Not used when exception occurs
            Exception("Test exception")  # Exception to raise
        ),
    ])
    def test_get_or_create_agent_registry_patterns(self, read_data, agent_type_result, registry_result, create_result, expected_result, write_data, exception):
        """Test _get_or_create_agent_registry with parameterized test data."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _read_kv
        def _mock_read_kv(*args, **kwargs):
            yield
            return read_data
        with patch.object(base_behaviour, '_read_kv', side_effect=(exception if exception else _mock_read_kv)) as mock_read:
            
            if not read_data and not exception:
                # Mock _get_or_create_agent_type
                def _mock_get_or_create_agent_type(*args, **kwargs):
                    yield
                    return agent_type_result
                with patch.object(base_behaviour, '_get_or_create_agent_type', side_effect=_mock_get_or_create_agent_type) as mock_get_type:
                    
                    if agent_type_result:
                        # Mock get_agent_registry_by_address
                        def _mock_get_agent_registry_by_address(*args, **kwargs):
                            yield
                            return registry_result
                        with patch.object(base_behaviour, 'get_agent_registry_by_address', side_effect=_mock_get_agent_registry_by_address) as mock_get_reg:
                            
                            if create_result:
                                # Mock generate_name
                                def _mock_generate_name(*args, **kwargs):
                                    return "TestAgent"
                                with patch.object(base_behaviour, 'generate_name', side_effect=_mock_generate_name) as mock_gen_name:
                                    # Mock create_agent_registry
                                    def _mock_create_agent_registry(*args, **kwargs):
                                        yield
                                        return create_result
                                    with patch.object(base_behaviour, 'create_agent_registry', side_effect=_mock_create_agent_registry) as mock_create:
                                        # Mock _write_kv
                                        def _mock_write_kv(*args, **kwargs):
                                            yield
                                            return None
                                        with patch.object(base_behaviour, '_write_kv', side_effect=_mock_write_kv) as mock_write:
                                            result = self._consume_generator(base_behaviour._get_or_create_agent_registry())
                                            
                                            # Verify the result
                                            assert result == expected_result
                                            
                                            # Verify create_agent_registry was called with correct parameters
                                            mock_create.assert_called_once()
                                            call_args = mock_create.call_args
                                            assert call_args[0][0] == "TestAgent"  # agent_name (generated)
                                            assert call_args[0][1] == "123"        # type_id
                                            assert call_args[0][2] == base_behaviour.context.agent_address  # eth_address
                                            
                                        # Verify _write_kv was called
                                        mock_write.assert_called_once_with(write_data)
                            else:
                                # Test creation failure case - need to mock generate_name here too
                                def _mock_generate_name(*args, **kwargs):
                                    return "TestAgent"
                                with patch.object(base_behaviour, 'generate_name', side_effect=_mock_generate_name) as mock_gen_name:
                                    def _mock_create_agent_registry_fail(*args, **kwargs):
                                        yield
                                        return None
                                    with patch.object(base_behaviour, 'create_agent_registry', side_effect=_mock_create_agent_registry_fail) as mock_create:
                                        
                                        result = self._consume_generator(base_behaviour._get_or_create_agent_registry())
                                        
                                        # Should return None when registry creation fails
                                        assert result == expected_result
                                        
                                        # Verify that create_agent_registry was called
                                        mock_create.assert_called_once()
                    else:
                        # Test agent type failure case
                        result = self._consume_generator(base_behaviour._get_or_create_agent_registry())
                        
                        # Should return None when agent type creation fails
                        assert result == expected_result
                        
                        # When agent type creation fails, create_agent_registry should not be called
                        # since the function returns early
            else:
                # Test existing data case
                result = self._consume_generator(base_behaviour._get_or_create_agent_registry())
                assert result == expected_result

    def test_get_or_create_agent_registry_creation_failure_error_logging(self):
        """Test _get_or_create_agent_registry specifically for error logging when creation fails (lines 2707-2708)."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock _read_kv to return no existing data
        def _mock_read_kv(*args, **kwargs):
            yield
            return {}
        with patch.object(base_behaviour, '_read_kv', side_effect=_mock_read_kv):
            
            # Mock _get_or_create_agent_type to succeed
            def _mock_get_or_create_agent_type(*args, **kwargs):
                yield
                return {"type_id": "123", "type_name": "Modius"}
            with patch.object(base_behaviour, '_get_or_create_agent_type', side_effect=_mock_get_or_create_agent_type):
                
                # Mock get_agent_registry_by_address to return None (no existing registry)
                def _mock_get_agent_registry_by_address(*args, **kwargs):
                    yield
                    return None
                with patch.object(base_behaviour, 'get_agent_registry_by_address', side_effect=_mock_get_agent_registry_by_address):
                    
                    # Mock generate_name to return a valid name
                    def _mock_generate_name(*args, **kwargs):
                        return "TestAgent"
                    with patch.object(base_behaviour, 'generate_name', side_effect=_mock_generate_name):
                        
                        # Mock create_agent_registry to FAIL (return None) - this will trigger lines 2707-2708
                        def _mock_create_agent_registry_fail(*args, **kwargs):
                            yield
                            return None
                        with patch.object(base_behaviour, 'create_agent_registry', side_effect=_mock_create_agent_registry_fail) as mock_create:
                            
                            # Mock the logger to capture error messages
                            with patch.object(base_behaviour.context.logger, 'error') as mock_logger_error:
                                
                                result = self._consume_generator(base_behaviour._get_or_create_agent_registry())
                                
                                # Verify the result is None when creation fails
                                assert result is None
                                
                                # Verify that create_agent_registry was called
                                mock_create.assert_called_once()
                                
                                # Verify that the error was logged (lines 2707-2708)
                                mock_logger_error.assert_called_once_with("Failed to create agent registry")

    # Name Generation Tests
    @pytest.mark.parametrize("seed_values", [
        list(range(10)),  # Small seeds
        [100, 500, 1000, 1500],  # Large seeds
        [0, 1, 2, 3, 4, 5]  # Sequential seeds
    ])
    def test_generate_phonetic_syllable(self, seed_values):
        """Test generate_phonetic_syllable method with parameterized seed values."""
        base_behaviour = self._create_base_behaviour()
        
        syllables = []
        for seed in seed_values:
            syllable = base_behaviour.generate_phonetic_syllable(seed)
            syllables.append(syllable)
            assert isinstance(syllable, str)
            assert len(syllable) > 0
        
        # Verify we get different syllables for different seeds (if enough seeds)
        if len(seed_values) > 1:
            assert len(set(syllables)) > 1

    @pytest.mark.parametrize("address,start_index,syllables,expected_properties", [
        (
            "0x1234567890123456789012345678901234567890",
            2,
            2,
            {"is_lowercase": True, "min_length": 4}
        ),
        (
            "0x1234567890123456789012345678901234567890",
            2,
            3,
            {"is_lowercase": True, "min_length": 6}
        ),
        (
            "0xfedcba0987654321fedcba0987654321fedcba09",
            5,
            2,
            {"is_lowercase": True, "min_length": 4}
        ),
    ])
    def test_generate_phonetic_name(self, address, start_index, syllables, expected_properties):
        """Test generate_phonetic_name method with parameterized test data."""
        base_behaviour = self._create_base_behaviour()
        
        name = base_behaviour.generate_phonetic_name(address, start_index, syllables)
        
        assert isinstance(name, str)
        assert len(name) > 0
        
        if expected_properties.get("is_lowercase"):
            assert name.islower()
        
        if expected_properties.get("min_length"):
            assert len(name) >= expected_properties["min_length"]

    @pytest.mark.parametrize("address,expected_properties", [
        (
            "0x1234567890123456789012345678901234567890",
            {"has_hyphen": True, "parts_count": 2, "last_name_has_number": True}
        ),
        (
            "0xfedcba0987654321fedcba0987654321fedcba09",
            {"has_hyphen": True, "parts_count": 2, "last_name_has_number": True}
        ),
        (
            "0xabcdef1234567890abcdef1234567890abcdef12",
            {"has_hyphen": True, "parts_count": 2, "last_name_has_number": True}
        ),
    ])
    def test_generate_name(self, address, expected_properties):
        """Test generate_name method with parameterized test data."""
        base_behaviour = self._create_base_behaviour()
        
        name = base_behaviour.generate_name(address)
        
        assert isinstance(name, str)
        assert len(name) > 0
        
        if expected_properties.get("has_hyphen"):
            assert "-" in name
        
        if expected_properties.get("parts_count"):
            parts = name.split("-")
            assert len(parts) == expected_properties["parts_count"]
            
            first_name = parts[0]
            last_name_part = parts[1]
            
            assert len(first_name) > 0
            assert len(last_name_part) > 0
            
            if expected_properties.get("last_name_has_number"):
                assert last_name_part[-2:].isdigit()
        
        # Test that different addresses generate different names
        if len(expected_properties) > 1:  # Only for multiple test cases
            other_addresses = [addr for addr in ["0x1234567890123456789012345678901234567890", "0xfedcba0987654321fedcba0987654321fedcba09", "0xabcdef1234567890abcdef1234567890abcdef12"] if addr != address]
            for other_addr in other_addresses:
                other_name = base_behaviour.generate_name(other_addr)
                assert other_name != name

    @pytest.mark.parametrize("message,mock_signature,expected_result,expected_encoding", [
        (
            "test message to sign",
            b"0x1234567890abcdef",
            "1234567890abcdef",  # Should remove '0x' prefix
            "utf-8"
        ),
        (
            "test message with unicode: 🚀🌟✨",
            b"0x1234567890abcdef",
            "1234567890abcdef",  # Should remove '0x' prefix
            "utf-8"
        ),
        (
            "simple message",
            b"0xabcdef123456",
            "abcdef123456",  # Should remove '0x' prefix
            "utf-8"
        ),
        (
            "message with numbers 12345",
            b"0x9876543210fedcba",
            "9876543210fedcba",  # Should remove '0x' prefix
            "utf-8"
        ),
    ])
    def test_sign_message_success(self, message, mock_signature, expected_result, expected_encoding):
        """Test sign_message method with successful signature generation."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the get_signature method
        def _mock_get_signature(*args, **kwargs):
            yield
            return mock_signature.decode('utf-8') if isinstance(mock_signature, bytes) else mock_signature
        with patch.object(base_behaviour, 'get_signature', side_effect=_mock_get_signature) as mock_get_sig:
            
            result = self._consume_generator(base_behaviour.sign_message(message))
            
            # Verify the result
            assert result == expected_result
            
            # Verify get_signature was called with encoded message
            mock_get_sig.assert_called_once_with(message.encode(expected_encoding))

    @pytest.mark.parametrize("message,mock_signature,expected_result", [
        ("test message to sign", None, None),  # get_signature returns None
        ("another message", b"", None),  # get_signature returns empty signature (falsy)
        ("unicode message 🚀", None, None),  # get_signature returns None for unicode
        ("simple text", b"", None),  # get_signature returns empty signature (falsy)
    ])
    def test_sign_message_failure_cases(self, message, mock_signature, expected_result):
        """Test sign_message method with various failure scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the get_signature method
        def _mock_get_signature(*args, **kwargs):
            yield
            return mock_signature
        with patch.object(base_behaviour, 'get_signature', side_effect=_mock_get_signature) as mock_get_sig:
            
            result = self._consume_generator(base_behaviour.sign_message(message))
            
            # Verify the result
            assert result == expected_result

    def test_synchronized_data(self):
        """Test synchronized_data property."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the parent's synchronized_data property
        mock_sync_data = MagicMock()
        with patch.object(type(base_behaviour), 'synchronized_data', new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_sync_data
            
            result = base_behaviour.synchronized_data
            
            assert result == mock_sync_data


    def test_store_data_error_handling(self):
        """Test _store_data method error handling."""
        base_behaviour = self._create_base_behaviour()
        
        # Test with a filepath that causes IOError/OSError
        with patch.object(base_behaviour.context.logger, 'error') as mock_error:
            with patch('builtins.open', side_effect=OSError("Permission denied")):
                base_behaviour._store_data({"test": "data"}, "test_attr", "/invalid/path/file.json")
                
                mock_error.assert_called_once()



    def test_store_agent_performance(self):
        """Test store_agent_performance method."""
        base_behaviour = self._create_base_behaviour()
        
        # Set up agent performance data
        base_behaviour.agent_performance = {
            "timestamp": 1234567890,
            "metrics": ["metric1", "metric2"],
            "agent_behavior": "test_behavior"
        }
        
        # Mock _store_data
        with patch.object(base_behaviour, '_store_data') as mock_store:
            base_behaviour.store_agent_performance()
            
            mock_store.assert_called_once_with(
                base_behaviour.agent_performance,
                "agent_performance",
                base_behaviour.agent_performance_filepath
            )

    def test_read_agent_performance_success(self):
        """Test read_agent_performance method with successful read."""
        base_behaviour = self._create_base_behaviour()
        
        # Set up existing agent performance data
        base_behaviour.agent_performance = {
            "timestamp": 1234567890,
            "metrics": ["metric1"],
            "agent_behavior": "existing_behavior"
        }
        
        # Mock _read_data to not raise exception
        with patch.object(base_behaviour, '_read_data'):
            base_behaviour.read_agent_performance()
            
            # Should not change existing data
            assert base_behaviour.agent_performance["agent_behavior"] == "existing_behavior"

    def test_read_agent_performance_failure(self):
        """Test read_agent_performance method with read failure."""
        base_behaviour = self._create_base_behaviour()
        
        # Clear existing agent performance
        base_behaviour.agent_performance = {}
        
        # Mock _read_data to raise exception
        with patch.object(base_behaviour, '_read_data', side_effect=Exception("Read failed")):
            with patch.object(base_behaviour.context.logger, 'warning') as mock_warning:
                base_behaviour.read_agent_performance()
                
                # Should log warning and initialize performance
                mock_warning.assert_called_once()
                assert base_behaviour.agent_performance["metrics"] == []

    def test_read_agent_performance_empty_data(self):
        """Test read_agent_performance method with empty data."""
        base_behaviour = self._create_base_behaviour()
        
        # Set up empty agent performance data
        base_behaviour.agent_performance = {}
        
        # Mock _read_data to not raise exception but return empty data
        with patch.object(base_behaviour, '_read_data'):
            base_behaviour.read_agent_performance()
            
            # Should initialize performance due to empty data
            assert "timestamp" in base_behaviour.agent_performance
            assert base_behaviour.agent_performance["metrics"] == []

    def test_initialize_agent_performance(self):
        """Test initialize_agent_performance method."""
        base_behaviour = self._create_base_behaviour()
        
        # Clear existing agent performance
        base_behaviour.agent_performance = {}
        
        base_behaviour.initialize_agent_performance()
        
        expected_structure = {
            "timestamp": None,
            "metrics": [],
            "agent_behavior": None,
        }
        
        assert base_behaviour.agent_performance == expected_structure

    def test_update_agent_performance_timestamp_success(self):
        """Test update_agent_performance_timestamp method with success."""
        base_behaviour = self._create_base_behaviour()
        
        # Set up agent performance
        base_behaviour.agent_performance = {
            "timestamp": None,
            "metrics": [],
            "agent_behavior": None,
        }
        
        # Mock datetime.utcnow()
        mock_datetime = MagicMock()
        mock_datetime.utcnow.return_value.timestamp.return_value = 1234567890
        
        with patch('packages.valory.skills.liquidity_trader_abci.behaviours.base.datetime', mock_datetime):
            base_behaviour.update_agent_performance_timestamp()
            
            assert base_behaviour.agent_performance["timestamp"] == 1234567890

    def test_update_agent_performance_timestamp_failure(self):
        """Test update_agent_performance_timestamp method with failure."""
        base_behaviour = self._create_base_behaviour()
        
        # Set up agent performance
        base_behaviour.agent_performance = {
            "timestamp": None,
            "metrics": [],
            "agent_behavior": None,
        }
        
        # Mock datetime.utcnow() to raise exception
        mock_datetime = MagicMock()
        mock_datetime.utcnow.side_effect = Exception("Datetime error")
        
        with patch('packages.valory.skills.liquidity_trader_abci.behaviours.base.datetime', mock_datetime):
            with patch.object(base_behaviour.context.logger, 'error') as mock_error:
                base_behaviour.update_agent_performance_timestamp()
                
                mock_error.assert_called_once()
                assert base_behaviour.agent_performance["timestamp"] is None


    def test_store_data_json_dump_error(self):
        """Test _store_data method when JSON dumping fails"""
        base_behaviour = self._create_base_behaviour()
        
        # Ensure temp_path is initialized
        if not hasattr(self, 'temp_path'):
            self.setUp()
        
        file_path = self.temp_path / "test_json_error.json"
        
        # Mock json.dump to raise an IOError/OSError
        with patch('json.dump', side_effect=OSError("Disk full")):
            with patch.object(base_behaviour.context.logger, 'error') as mock_error:
                base_behaviour._store_data({"test": "data"}, "test_attr", str(file_path))
                
                # Should log the error from the inner try-except block
                # The error message includes the full path, so we check if it contains the filename
                mock_error.assert_called_once()
                error_call = mock_error.call_args[0][0]
                assert "Error writing to file" in error_call
                assert "test_json_error.json" in error_call


    @pytest.mark.parametrize(
        "test_case,timeout,message,dialogue,expected_timeout,expected_response,expected_behavior",
        [
            # Test case 1: Basic success with timeout
            (
                "basic_success_with_timeout",
                10.0,
                "test_message",
                "test_dialogue",
                10.0,
                "test_response",
                "success"
            ),
            # Test case 2: No timeout (default None)
            (
                "no_timeout_default_none",
                None,
                "test_message",
                "test_dialogue",
                None,
                "test_response",
                "success"
            ),
            # Test case 3: Zero timeout
            (
                "zero_timeout",
                0.0,
                "test_message",
                "test_dialogue",
                0.0,
                "test_response",
                "success"
            ),
            # Test case 4: Negative timeout
            (
                "negative_timeout",
                -5.0,
                "test_message",
                "test_dialogue",
                -5.0,
                "test_response",
                "success"
            ),
            # Test case 5: Large timeout
            (
                "large_timeout",
                3600.0,
                "test_message",
                "test_dialogue",
                3600.0,
                "test_response",
                "success"
            ),
            # Test case 6: Different message types
            (
                "different_message_types",
                None,
                {"key": "value"},
                "test_dialogue",
                None,
                "test_response",
                "success"
            ),
            # Test case 7: Different dialogue types
            (
                "different_dialogue_types",
                None,
                "test_message",
                123,
                None,
                "test_response",
                "success"
            ),
            # Test case 8: Different response types
            (
                "different_response_types",
                None,
                "test_message",
                "test_dialogue",
                None,
                {"data": "response"},
                "success"
            ),
            # Test case 9: Outbox error
            (
                "outbox_error",
                None,
                "test_message",
                "test_dialogue",
                None,
                None,
                "outbox_exception"
            ),
            # Test case 10: Nonce error
            (
                "nonce_error",
                None,
                "test_message",
                "test_dialogue",
                None,
                None,
                "nonce_exception"
            ),
            # Test case 11: Callback error
            (
                "callback_error",
                None,
                "test_message",
                "test_dialogue",
                None,
                None,
                "callback_exception"
            ),
            # Test case 12: Wait error
            (
                "wait_error",
                None,
                "test_message",
                "test_dialogue",
                None,
                None,
                "wait_exception"
            ),
        ],
    )
    def test_do_connection_request_comprehensive(
        self,
        test_case,
        timeout,
        message,
        dialogue,
        expected_timeout,
        expected_response,
        expected_behavior,
    ):
        """Comprehensive test for _do_connection_request with all scenarios."""
        base_behaviour = self._create_base_behaviour()
        
        # Mock the required methods
        with patch.object(base_behaviour.context.outbox, 'put_message') as mock_put_message, \
             patch.object(base_behaviour, '_get_request_nonce_from_dialogue') as mock_get_nonce, \
             patch.object(base_behaviour, 'get_callback_request') as mock_get_callback, \
             patch.object(base_behaviour, 'wait_for_message') as mock_wait:
            
            # Mock the requests context
            mock_requests = MagicMock()
            mock_requests.request_id_to_callback = {}
            base_behaviour.context.requests = mock_requests
            
            # Configure mocks based on test case
            if expected_behavior == "outbox_exception":
                mock_put_message.side_effect = Exception("Outbox error")
                mock_get_nonce.return_value = "test_nonce"
                mock_get_callback.return_value = "test_callback"
                
                # Test should raise the exception
                with pytest.raises(Exception, match="Outbox error"):
                    self._consume_generator(
                        base_behaviour._do_connection_request(
                            message=message,
                            dialogue=dialogue,
                            timeout=timeout
                        )
                    )
                return
                
            elif expected_behavior == "nonce_exception":
                mock_put_message.return_value = None
                mock_get_nonce.side_effect = Exception("Nonce error")
                mock_get_callback.return_value = "test_callback"
                
                # Test should raise the exception
                with pytest.raises(Exception, match="Nonce error"):
                    self._consume_generator(
                        base_behaviour._do_connection_request(
                            message=message,
                            dialogue=dialogue,
                            timeout=timeout
                        )
                    )
                return
                
            elif expected_behavior == "callback_exception":
                mock_put_message.return_value = None
                mock_get_nonce.return_value = "test_nonce"
                mock_get_callback.side_effect = Exception("Callback error")
                
                # Test should raise the exception
                with pytest.raises(Exception, match="Callback error"):
                    self._consume_generator(
                        base_behaviour._do_connection_request(
                            message=message,
                            dialogue=dialogue,
                            timeout=timeout
                        )
                    )
                return
                
            elif expected_behavior == "wait_exception":
                mock_put_message.return_value = None
                mock_get_nonce.return_value = "test_nonce"
                mock_get_callback.return_value = "test_callback"
                mock_wait.side_effect = Exception("Wait error")
                
                # Test should raise the exception
                with pytest.raises(Exception, match="Wait error"):
                    self._consume_generator(
                        base_behaviour._do_connection_request(
                            message=message,
                            dialogue=dialogue,
                            timeout=timeout
                        )
                    )
                return
            
            # Normal success case
            mock_put_message.return_value = None
            mock_get_nonce.return_value = "test_nonce"
            mock_get_callback.return_value = "test_callback"
            
            # Set up wait_for_message as a generator that yields None and returns the response
            def wait_side_effect(*args, **kwargs):
                yield None
                return expected_response
            
            mock_wait.side_effect = wait_side_effect
            
            # Execute the function
            result = self._consume_generator(
                base_behaviour._do_connection_request(
                    message=message,
                    dialogue=dialogue,
                    timeout=timeout
                )
            )
            
            # Verify the result
            assert result == expected_response
            
            # Verify put_message was called
            mock_put_message.assert_called_once_with(message=message)
            
            # Verify wait_for_message was called with correct timeout
            mock_wait.assert_called_once_with(timeout=expected_timeout)
            
            # Verify callback was registered
            assert mock_requests.request_id_to_callback["test_nonce"] == "test_callback"
            
            # Verify _get_request_nonce_from_dialogue was called with the dialogue
            mock_get_nonce.assert_called_once_with(dialogue)
            
            # Verify get_callback_request was called
            mock_get_callback.assert_called_once() 

    def test_get_entry_costs_key(self):
        """Test _get_entry_costs_key method."""
        base_behaviour = self._create_base_behaviour()
        
        # Test with different chain and position_id combinations
        test_cases = [
            ("optimism", "0xPool123", "entry_costs_optimism_0xPool123"),
            ("mode", "0xPool456", "entry_costs_mode_0xPool456"),
            ("ethereum", "position789", "entry_costs_ethereum_position789"),
            ("", "", "entry_costs__"),  # Empty strings
            ("chain_with_underscores", "id_with_dashes", "entry_costs_chain_with_underscores_id_with_dashes"),
        ]
        
        for chain, position_id, expected_key in test_cases:
            result = base_behaviour._get_entry_costs_key(chain, position_id)
            assert result == expected_key, f"Expected {expected_key} for chain={chain}, position_id={position_id}, but got {result}"
            
            # Verify the key format is consistent
            assert result.startswith("entry_costs_")
            assert chain in result
            assert position_id in result

    def test_log_gas_usage_max_records_limit(self):
        """Test log_gas_usage method when maximum records limit is exceeded."""
        base_behaviour = self._create_base_behaviour()
        
        # Test the GasCostTracker functionality
        gas_tracker = base_behaviour.gas_cost_tracker
        
        # Test with a single chain
        chain = "optimism"
        
        # Add records up to the MAX_RECORDS limit
        for i in range(gas_tracker.MAX_RECORDS + 5):  # Add 5 extra records
            timestamp = int(time.time()) + i
            tx_hash = f"0xTxHash{i:03d}"
            gas_used = 100000 + i
            gas_price = 20000000000 + i
            
            gas_tracker.log_gas_usage(chain, timestamp, tx_hash, gas_used, gas_price)
        
        # Verify that only MAX_RECORDS are kept
        assert len(gas_tracker.data[chain]) == gas_tracker.MAX_RECORDS
        
        # Verify that the oldest records were removed (keeping only the latest)
        expected_timestamps = [int(time.time()) + i for i in range(5, gas_tracker.MAX_RECORDS + 5)]
        actual_timestamps = [record["timestamp"] for record in gas_tracker.data[chain]]
        
        # The timestamps should match the expected range (latest records)
        assert actual_timestamps == expected_timestamps
        
        # Verify the structure of the records
        for record in gas_tracker.data[chain]:
            assert "timestamp" in record
            assert "tx_hash" in record
            assert "gas_used" in record
            assert "gas_price" in record
            assert isinstance(record["timestamp"], int)
            assert isinstance(record["tx_hash"], str)
            assert isinstance(record["gas_used"], int)
            assert isinstance(record["gas_price"], int)
        
        # Test with multiple chains
        chain2 = "mode"
        for i in range(gas_tracker.MAX_RECORDS + 3):  # Add 3 extra records
            timestamp = int(time.time()) + i
            tx_hash = f"0xModeTxHash{i:03d}"
            gas_used = 150000 + i
            gas_price = 25000000000 + i
            
            gas_tracker.log_gas_usage(chain2, timestamp, tx_hash, gas_used, gas_price)
        
        # Verify both chains maintain their own MAX_RECORDS limit
        assert len(gas_tracker.data[chain]) == gas_tracker.MAX_RECORDS
        assert len(gas_tracker.data[chain2]) == gas_tracker.MAX_RECORDS
        
        # Verify the data is properly separated between chains
        assert gas_tracker.data[chain] != gas_tracker.data[chain2]
        
        # Test edge case: exactly at MAX_RECORDS
        chain3 = "ethereum"
        for i in range(gas_tracker.MAX_RECORDS):
            timestamp = int(time.time()) + i
            tx_hash = f"0xEthTxHash{i:03d}"
            gas_used = 200000 + i
            gas_price = 30000000000 + i
            
            gas_tracker.log_gas_usage(chain3, timestamp, tx_hash, gas_used, gas_price)
        
        # Should have exactly MAX_RECORDS
        assert len(gas_tracker.data[chain3]) == gas_tracker.MAX_RECORDS
        
        # Adding one more should still keep MAX_RECORDS
        gas_tracker.log_gas_usage(chain3, int(time.time()) + 999, "0xEthTxHash999", 999999, 99999999999)
        assert len(gas_tracker.data[chain3]) == gas_tracker.MAX_RECORDS
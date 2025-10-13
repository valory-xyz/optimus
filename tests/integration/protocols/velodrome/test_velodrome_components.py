# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2025 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
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

"""Unit integration tests for Velodrome protocol components."""

import pytest
from unittest.mock import MagicMock

from packages.valory.skills.liquidity_trader_abci.pools.velodrome import VelodromePoolBehaviour

from tests.integration.protocols.base.protocol_test_base import ProtocolIntegrationTestBase


class TestVelodromeComponents(ProtocolIntegrationTestBase):
    """Test individual Velodrome protocol components in isolation."""

    def test_velodrome_abstract_class_cannot_be_instantiated(self):
        """Test that VelodromePoolBehaviour is abstract and cannot be instantiated."""
        # This test verifies that VelodromePoolBehaviour is indeed abstract
        # and cannot be instantiated directly due to missing _get_tokens implementation
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VelodromePoolBehaviour(
                name="TestBehaviour",
                skill_context=MagicMock(),
                params=MagicMock()
            )

    def test_velodrome_methods_exist(self):
        """Test that VelodromePoolBehaviour has the expected methods."""
        # Test that the class has the methods we expect
        assert hasattr(VelodromePoolBehaviour, 'enter')
        assert hasattr(VelodromePoolBehaviour, 'exit')
        assert hasattr(VelodromePoolBehaviour, 'stake_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'unstake_lp_tokens')
        assert hasattr(VelodromePoolBehaviour, 'claim_rewards')
        assert hasattr(VelodromePoolBehaviour, 'get_pending_rewards')
        assert hasattr(VelodromePoolBehaviour, 'get_staked_balance')
        assert hasattr(VelodromePoolBehaviour, 'get_gauge_address')
        assert hasattr(VelodromePoolBehaviour, '_get_pool_tokens')
        assert hasattr(VelodromePoolBehaviour, 'calculate_tick_range_from_bands_wrapper')
        assert hasattr(VelodromePoolBehaviour, 'optimize_stablecoin_bands')
        assert hasattr(VelodromePoolBehaviour, 'get_velodrome_amounts_for_liquidity')

    def test_velodrome_missing_get_tokens_method(self):
        """Test that VelodromePoolBehaviour is missing the _get_tokens implementation."""
        # This test confirms that _get_tokens is not implemented in VelodromePoolBehaviour
        # which is why it cannot be instantiated
        
        # Check if _get_tokens is implemented (it should not be)
        import inspect
        source = inspect.getsource(VelodromePoolBehaviour._get_tokens)
        assert "pass" in source or "raise NotImplementedError" in source

    def test_velodrome_method_signatures(self):
        """Test that VelodromePoolBehaviour methods have correct signatures."""
        import inspect
        
        # Test enter method signature - uses **kwargs pattern
        enter_sig = inspect.signature(VelodromePoolBehaviour.enter)
        assert 'kwargs' in enter_sig.parameters
        assert enter_sig.parameters['kwargs'].kind == inspect.Parameter.VAR_KEYWORD
        
        # Test exit method signature - uses **kwargs pattern  
        exit_sig = inspect.signature(VelodromePoolBehaviour.exit)
        assert 'kwargs' in exit_sig.parameters
        assert exit_sig.parameters['kwargs'].kind == inspect.Parameter.VAR_KEYWORD
        
        # Test that methods return generators
        assert enter_sig.return_annotation.__name__ == 'Generator'
        assert exit_sig.return_annotation.__name__ == 'Generator'

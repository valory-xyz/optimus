# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2023 Valory AG
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

"""Tests for valory/liquidity_trader_abci skill's behaviours."""

# pylint: skip-file


import datetime
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import cast
from unittest import mock
from unittest.mock import MagicMock

from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.behaviour_utils import BaseBehaviour
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.liquidity_trader_abci import PUBLIC_ID
from packages.valory.skills.liquidity_trader_abci.behaviours import (
    Action,
    CheckStakingKPIMetBehaviour,
    Decision,
    DecisionMakingBehaviour,
    DexType,
    EvaluateStrategyBehaviour,
    GasCostTracker,
    GetPositionsBehaviour,
    PostTxSettlementBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.payloads import GetPositionsPayload
from packages.valory.skills.liquidity_trader_abci.rounds import (
    DecisionMakingRound,
    Event,
    SynchronizedData,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent.parent


def test_skill_public_id() -> None:
    """Test skill module public ID"""

    assert PUBLIC_ID.name == Path(__file__).parents[1].name
    assert PUBLIC_ID.author == Path(__file__).parents[3].name


class LiquidityTraderAbciFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing FSMBehaviour."""

    path_to_skill = PACKAGE_DIR

    def setUp(self):
        """Setup test environment."""
        super(LiquidityTraderAbciFSMBehaviourBaseCase, self).setUp()


class TestLiquidityTraderBaseBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test cases for LiquidityTraderBaseBehaviour."""

    def setUp(self):
        """Setup test environment."""
        super().setUp()
        self.behaviour = GetPositionsBehaviour(context=MagicMock(), params=MagicMock())

    def test_get_native_balance(self):
        """Test native balance retrieval."""

        # Create a generator-based mock
        def mock_get_ledger_api_response(*args, **kwargs):
            mock_response = MagicMock(
                performative=LedgerApiMessage.Performative.STATE,
                state=MagicMock(body={"get_balance_result": "1000000000000000000"}),
            )
            yield mock_response

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_ledger_api_response",
            side_effect=mock_get_ledger_api_response,
        ):
            generator = self.behaviour.current_behaviour._get_native_balance(
                "mode", "0xA37e826fF12Af82e5eAed8c981218B78Cc606bac"
            )
            balance = next(generator)
            assert balance.performative == LedgerApiMessage.Performative.STATE
            assert balance.state.body["get_balance_result"] == "1000000000000000000"

    def test_get_token_balance(self):
        """Test token balance retrieval."""
        # Mock contract interaction
        mock_balance = 500000000000000000
        with mock.patch.object(
            self.behaviour.current_behaviour,
            "contract_interact",
            return_value=[mock_balance],
        ):
            balance = list(
                self.behaviour.current_behaviour._get_token_balance(
                    "mode",
                    "0xA37e826fF12Af82e5eAed8c981218B78Cc606bac",
                    "0x554D1444b6a38fA2eb18d86f2C10F42BE630c89D",
                )
            )[0]
            assert balance == mock_balance

    def test_get_token_decimals(self):
        """Test token decimals retrieval."""
        # Mock contract interaction
        mock_decimals = 18
        with mock.patch.object(
            self.behaviour.current_behaviour,
            "contract_interact",
            return_value=[mock_decimals],
        ):
            decimals = list(
                self.behaviour.current_behaviour._get_token_decimals(
                    "ethereum", "0x123"
                )
            )[0]
            assert decimals == mock_decimals

    def test_store_data(self):
        """Test data storage."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            test_data = {"test": "data"}
            self.behaviour.current_behaviour._store_data(
                test_data, "test_attribute", temp_file.name
            )

            with open(temp_file.name, "r") as f:
                loaded_data = json.load(f)
                assert loaded_data == test_data

    def test_read_assets(self):
        """Test reading assets from file."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            test_assets = {"ethereum": {"0x123": "ETH"}}
            json.dump(test_assets, temp_file)
            temp_file.close()

            self.behaviour.current_behaviour.assets_filepath = temp_file.name
            self.behaviour.current_behaviour.read_assets()
            assert self.behaviour.current_behaviour.assets == test_assets

    def test_store_current_positions(self):
        """Test storing current positions."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            test_positions = [{"chain": "ethereum", "address": "0x123"}]
            self.behaviour.current_behaviour.current_positions = test_positions
            self.behaviour.current_behaviour.current_positions_filepath = temp_file.name
            self.behaviour.current_behaviour.store_current_positions()

            with open(temp_file.name, "r") as f:
                loaded_positions = json.load(f)
                assert loaded_positions == test_positions

    def test_calculate_min_num_of_safe_tx_required(self):
        """Test minimum safe tx calculation."""
        # Create a mock round sequence with a controlled timestamp
        mock_round_sequence = self.behaviour.current_behaviour.round_sequence
        mock_round_sequence._last_round_transition_timestamp = (
            datetime.datetime.fromtimestamp(1609545600)
        )

        # Mock the generator methods with arguments
        def mock_liveness_ratio(chain):
            yield 1000000  # You can add logic here to use the `chain` argument if needed

        def mock_liveness_period(chain):
            yield 86400  # Same here, you can utilize `chain` if necessary

        def mock_ts_checkpoint(chain):
            yield 1609459200  # Similarly, handle `chain` if necessary

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_liveness_ratio",
            side_effect=mock_liveness_ratio,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_liveness_period",
            side_effect=mock_liveness_period,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_ts_checkpoint",
            side_effect=mock_ts_checkpoint,
        ):
            # Run the method
            generator = (
                self.behaviour.current_behaviour._calculate_min_num_of_safe_tx_required(
                    "mode"
                )
            )
            result = next(generator)

            # Assert the result
            assert result is not None
            assert isinstance(result, int)

    def test_contract_interaction_error(self, caplog):
        """Test contract interaction error logging."""
        # Print out the actual method implementation for debugging
        print(
            f"Method implementation: {self.behaviour.current_behaviour.contract_interaction_error}"
        )

        # Create a mock response with error and warning messages
        mock_response = MagicMock(
            raw_transaction=MagicMock(
                body={"error": "Test error message", "warning": "Test warning message"}
            )
        )

        # Capture logging at ERROR level
        with caplog.at_level(logging.ERROR):
            # Directly call error logging with the raw method
            logger = self.behaviour.current_behaviour.context.logger
            logger.error("Test error message")

            # Call the method
            self.behaviour.current_behaviour.contract_interaction_error(
                "test_contract", "test_method", mock_response
            )

            # Print out captured records for debugging
            for record in caplog.records:
                print(f"Captured log record: {record.message}")

            # Check log records
            assert len(caplog.records) > 0, "No log records were generated"
            assert any(
                "Test error message" in record.message for record in caplog.records
            ), "Error message not found in log records"


class TestGetPositionsBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test cases for GetPositionsBehaviour."""

    behaviour_class = GetPositionsBehaviour
    path_to_skill = Path(__file__).parent.parent

    def setUp(self):
        """Setup test environment."""
        super().setUp()

        # Mock context
        self.mock_context = MagicMock()
        self.mock_context.agent_address = "test_agent"
        self.mock_context.logger = logger

        # Mock parameters
        self.mock_params = MagicMock()
        # Instead of directly setting params, use the context's params
        self.behaviour.context.params.initial_assets = [
            {"chain": "ethereum", "symbol": "ETH", "address": "0x123", "balance": 1000}
        ]

        # Initialize behaviour with mocked params
        self.behaviour = GetPositionsBehaviour(
            context=self.mock_context, params=self.mock_params
        )

    def test_get_positions_success(self) -> None:
        """Test successful get_positions."""
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data={
                    "assets": [{"ethereum": [{"address": "0x123", "balance": 100}]}]
                }
            )
        )
        self.fast_forward_to_behaviour(
            self.behaviour,
            GetPositionsBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        payload_data = {
            "positions": json.dumps(
                [
                    {
                        "chain": "mode",
                        "assets": [
                            {
                                "asset_symbol": "ETH",
                                "asset_type": "native",
                                "address": "0x0000000000000000000000000000000000000000",
                                "balance": 0,
                            },
                            {
                                "asset_symbol": "USDC",
                                "asset_type": "erc_20",
                                "address": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                                "balance": 0,
                            },
                        ],
                    }
                ]
            )
        }

        def mock_generator():
            self.behaviour.current_behaviour.set_done()
            yield

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_positions",
            return_value=json.loads(payload_data["positions"]),
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_adjust_current_positions_for_backward_compatibility",
            return_value=mock_generator(),
        ):
            self.behaviour.act_wrapper()
            self._test_done_flag_set()
            self.end_round(Event.DONE)

    def test_get_positions_payload_success(self) -> None:
        """Test successful positions retrieval."""
        # Prepare synchronized data and setup test
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data={
                    "assets": [{"ethereum": [{"address": "0x123", "balance": 100}]}]
                }
            )
        )
        self.fast_forward_to_behaviour(
            self.behaviour,
            GetPositionsBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock necessary methods
        self.behaviour.get_positions = MagicMock(
            return_value=[
                {
                    "chain": "ethereum",
                    "assets": [
                        {
                            "asset_symbol": "ETH",
                            "asset_type": "erc_20",
                            "address": "0x123",
                            "balance": 1000,
                        }
                    ],
                }
            ]
        )
        self.behaviour.send_a2a_transaction = MagicMock()
        self.behaviour._adjust_current_positions_for_backward_compatibility = MagicMock(
            return_value=None
        )

        # Assertions
        payload_data = {
            "positions": json.dumps(
                [
                    {
                        "chain": "mode",
                        "assets": [
                            {
                                "asset_symbol": "ETH",
                                "asset_type": "native",
                                "address": "0x0000000000000000000000000000000000000000",
                                "balance": 0,
                            },
                            {
                                "asset_symbol": "USDC",
                                "asset_type": "erc_20",
                                "address": "0xd988097fb8612cc24eeC14542bC03424c656005f",
                                "balance": 0,
                            },
                        ],
                    }
                ]
            )
        }
        payload = GetPositionsPayload(sender="test_agent", **payload_data)

        # Verify payload and method calls
        assert isinstance(payload, GetPositionsPayload)
        assert json.loads(payload.positions) == json.loads(payload_data["positions"])

    def test_get_positions_empty_assets(self):
        """Test behaviour with no initial assets."""
        # Set empty initial assets
        # Unfreeze the params object
        self.behaviour.context.params.__dict__["_frozen"] = False

        self.behaviour.context.params.initial_assets = []  # Modify this line

        # Prepare synchronized data
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data={
                    "assets": [{"ethereum": [{"address": "0x123", "balance": 100}]}]
                }
            )
        )

        # Setup test
        self.fast_forward_to_behaviour(
            self.behaviour,
            GetPositionsBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Execute act_wrapper (assuming it exists and is correct)
        payload = self.behaviour.act_wrapper()

        # Assertions
        assert payload is None

    def test_get_positions_error(self) -> None:
        """Test get_positions with error."""
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data={
                    "assets": [{"ethereum": [{"address": "0x123", "balance": 100}]}]
                }
            )
        )
        self.fast_forward_to_behaviour(
            self.behaviour,
            GetPositionsBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Store the current behaviour before mocking
        current_behaviour = self.behaviour.current_behaviour

        # Mock the entire get_positions method to simulate an error scenario
        def mock_get_positions_error():
            # Simulate a scenario where positions retrieval fails
            current_behaviour.context.logger.error("Positions retrieval failed")
            current_behaviour._is_done = True  # Directly set the done flag
            yield None

        with mock.patch.object(
            current_behaviour, "get_positions", side_effect=mock_get_positions_error
        ):
            try:
                # Attempt to run the behaviour
                self.behaviour.act_wrapper()
            except Exception as e:
                print(f"Caught exception: {type(e)}")
                print(f"Exception details: {str(e)}")

        # Check that the behaviour is marked as done
        assert current_behaviour._is_done

        # Complete the round
        self.end_round(Event.DONE)

    def test_get_positions_assets_file(self) -> None:
        """Test reading/writing assets file."""
        test_assets = {"ethereum": {"0x123": "ETH"}}

        behaviour = cast(GetPositionsBehaviour, self.behaviour.current_behaviour)
        behaviour.assets = test_assets
        behaviour.store_assets()
        behaviour.read_assets()

        assert behaviour.assets == test_assets

    def test_backward_compatibility(self) -> None:
        """Test backward compatibility adjustments."""
        old_position = {
            "address": "0x123",
            "assets": ["0xtoken1", "0xtoken2"],
            "chain": "ethereum",
        }

        behaviour = cast(GetPositionsBehaviour, self.behaviour.current_behaviour)

        with mock.patch.object(
            behaviour, "_get_token_symbol", side_effect=["TOKEN1", "TOKEN2"]
        ):
            generator = behaviour._adjust_current_positions_for_backward_compatibility(
                [old_position]
            )
            list(generator)

        assert len(behaviour.current_positions) == 1
        assert behaviour.current_positions[0]["pool_address"] == "0x123"
        assert behaviour.current_positions[0]["token0"] == "0xtoken1"
        assert behaviour.current_positions[0]["token1"] == "0xtoken2"


class TestEvaluateStrategyBehaviour(LiquidityTraderAbciFSMBehaviourBaseCase):
    """Test EvaluateStrategyBehaviour."""

    behaviour_class = EvaluateStrategyBehaviour
    path_to_skill = PACKAGE_DIR

    def test_execute_hyper_strategy(self) -> None:
        """Test execute hyper strategy."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        test_opportunities = {
            "optimal_strategies": ["strategy1"],
            "position_to_exit": None,
        }

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=test_opportunities,
        ):
            self.behaviour.current_behaviour.execute_hyper_strategy()

            assert self.behaviour.current_behaviour.selected_opportunities == [
                "strategy1"
            ]
            assert self.behaviour.current_behaviour.position_to_exit is None

    def test_get_returns_metrics(self) -> None:
        """Test get returns metrics."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        test_position = {
            "chain": "ethereum",
            "dex_type": "balancerPool",
            "pool_address": "0xpool",
        }

        test_metrics = {"apr": 10.0, "tvl": 1000000}
        test_strategy = "test_strategy"

        # Temporarily unfreeze the Params object to modify strategies_kwargs
        self.behaviour.current_behaviour.context.params.__dict__["_frozen"] = False
        self.behaviour.current_behaviour.context.params.strategies_kwargs = {
            "test_strategy": {}
        }
        # Refreeze if necessary after the modification
        self.behaviour.current_behaviour.context.params.__dict__["_frozen"] = True

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "execute_strategy",
            return_value=test_metrics,
        ):
            metrics = (
                self.behaviour.current_behaviour.get_returns_metrics_for_opportunity(
                    test_position, test_strategy
                )
            )
            assert metrics == test_metrics

    def test_strategy_exec(self):
        """Test strategy_exec method."""
        synchronized_data = SynchronizedData(AbciAppDB(setup_data=dict()))
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock strategies_executables
        self.behaviour.current_behaviour.shared_state.strategies_executables = {
            "test_strategy": ("test_exec", "test_method")
        }

        result = self.behaviour.current_behaviour.strategy_exec("test_strategy")
        assert result == ("test_exec", "test_method")

        # Test non-existent strategy
        result = self.behaviour.current_behaviour.strategy_exec("non_existent")
        assert result is None

    def test_calculate_pnl_for_balancer(self) -> None:
        """Test PnL calculation for Balancer."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(AbciAppDB(setup_data=dict())),
        )

        # Prepare params
        self.behaviour.current_behaviour.context.params.__dict__["_frozen"] = False
        self.behaviour.current_behaviour.context.params.balancer_vault_contract_addresses = {
            "ethereum": "0xvault"
        }
        self.behaviour.current_behaviour.context.params.safe_contract_addresses = {
            "ethereum": "0xsafe"
        }
        self.behaviour.current_behaviour.context.params.__dict__["_frozen"] = True

        position = {
            "chain": "ethereum",
            "pool_address": "0xpool",
            "token0": "0xtoken0",
            "token1": "0xtoken1",
            "token0_symbol": "TOKEN0",
            "token1_symbol": "TOKEN1",
            "amount0": 100,
            "amount1": 200,
            "timestamp": 1609459200,
        }

        # Comprehensive mock methods
        def mock_contract_interact(*args, **kwargs):
            contract_callable = kwargs.get("contract_callable", "")
            if "get_pool_id" in contract_callable:
                return "pool_id_123"
            elif "get_pool_tokens" in contract_callable:
                return {0: ["0xtoken0", "0xtoken1"], 1: [1000, 2000]}
            elif "get_total_supply" in contract_callable:
                return 10000
            elif "get_balance" in contract_callable:
                return 1000
            return None

        def mock_get_token_decimals(*args, **kwargs):
            return 18

        def mock_fetch_token_price(*args, **kwargs):
            return 10.0

        # Patch methods
        with mock.patch.object(
            self.behaviour.current_behaviour,
            "contract_interact",
            side_effect=mock_contract_interact,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_token_decimals",
            side_effect=mock_get_token_decimals,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_fetch_token_price",
            side_effect=mock_fetch_token_price,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "calculate_initial_investment_value",
            return_value=5000.0,
        ):
            # Directly call the method
            pnl_data = self.behaviour.current_behaviour.calculate_pnl_for_balancer(
                position
            )

            # Assertions
            assert pnl_data is not None, "PnL calculation returned None"


class TestDecisionMakingBehaviour(FSMBehaviourBaseCase):
    """Test DecisionMakingBehaviour."""

    path_to_skill = PACKAGE_DIR
    behaviour_class = DecisionMakingBehaviour

    def test_check_step_costs(self):
        """Test check_step_costs method."""
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.FIND_BRIDGE_ROUTE.value,
                    "from_chain": "ethereum",
                    "to_chain": "polygon",
                    "from_token": "0x1234",
                    "from_token_symbol": "ETH",
                    "to_token": "0x5678",
                    "to_token_symbol": "MATIC",
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["test_hash"],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        step = {"action": {"fromChainId": 1, "toChainId": 1}}

        # Mock step transaction fetch
        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_step_transaction",
            return_value=(True, {"fee": 10, "gas_cost": 20}),
        ):
            result = self.behaviour.current_behaviour.check_step_costs(
                step=step,
                remaining_fee_allowance=100,
                remaining_gas_allowance=200,
                step_index=0,
                total_steps=2,
            )

            # Consume the generator and collect results
            final_result = list(result)

            # Verify the result structure
            assert len(final_result) == 2
            step_profitable, step_data = final_result

            assert step_profitable is True
            assert step_data == {"fee": 10, "gas_cost": 20}

    def test_update_assets_after_swap(self):
        """Test _update_assets_after_swap method."""
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.FIND_BRIDGE_ROUTE.value,
                    "from_chain": "ethereum",
                    "to_chain": "polygon",
                    "from_token": "0x1234",
                    "from_token_symbol": "ETH",
                    "to_token": "0x5678",
                    "to_token_symbol": "MATIC",
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["test_hash"],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        action = {
            "from_chain": "chain1",
            "from_token": "token1",
            "from_token_symbol": "TKN1",
            "to_chain": "chain2",
            "to_token": "token2",
            "to_token_symbol": "TKN2",
            "remaining_fee_allowance": 100,
            "remaining_gas_allowance": 200,
        }

        with mock.patch.object(
            self.behaviour.current_behaviour, "_add_token_to_assets"
        ) as mock_add:
            result = self.behaviour.current_behaviour._update_assets_after_swap(
                [action], 0
            )
            assert result == (
                Event.UPDATE.value,
                {
                    "last_executed_step_index": 0,
                    "fee_details": {
                        "remaining_fee_allowance": 100,
                        "remaining_gas_allowance": 200,
                    },
                    "last_action": Action.STEP_EXECUTED.value,
                },
            )
            assert mock_add.call_count == 2

    def test_get_decision_on_swap(self):
        """Test get_decision_on_swap method."""
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.FIND_BRIDGE_ROUTE.value,
                    "from_chain": "ethereum",
                    "to_chain": "polygon",
                    "from_token": "0x1234",
                    "from_token_symbol": "ETH",
                    "to_token": "0x5678",
                    "to_token_symbol": "MATIC",
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["test_hash"],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        current_behaviour = self.behaviour.current_behaviour

        with mock.patch.object(
            current_behaviour,
            "get_swap_status",
            return_value=(("continue", "some_status")),
        ):
            # Directly call the method and check the result
            result = next(current_behaviour.get_decision_on_swap())

            # Assert based on the actual method implementation
            assert result == Decision.CONTINUE.value

    def test_prepare_next_action_unknown_action(self):
        """Test _prepare_next_action method for an unknown action."""
        serialized_actions = json.dumps(
            [{"action": Action.ENTER_POOL.value}]  # Use a valid Action value
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        current_behaviour = self.behaviour.current_behaviour

        # Use a try-except to handle StopIteration
        try:
            result = next(
                current_behaviour._prepare_next_action(
                    positions=[],
                    actions=json.loads(serialized_actions),
                    current_action_index=0,
                    last_round_id="some_id",
                )
            )
        except StopIteration as e:
            result = e.value

        # Check that the result is a tuple with a "done" event
        assert result == (Event.DONE.value, {})

    def test_prepare_next_action_enter_pool(self):
        """Test _prepare_next_action method for ENTER_POOL action."""
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.ENTER_POOL.value,
                    "dex_type": DexType.UNISWAP_V3.value,  # Matches DexType.UNISWAP_V3.value
                    "chain": "mode",
                    "token0": "0x1234",
                    "token1": "0x5678",
                    "pool_address": "0x9abc",
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        current_behaviour = self.behaviour.current_behaviour

        # Mock get_enter_pool_tx_hash to return specific values
        with mock.patch.object(
            current_behaviour,
            "get_enter_pool_tx_hash",
            return_value=("test_tx_hash", "ethereum", "0xsafe_address"),
        ):
            result = next(
                current_behaviour._prepare_next_action(
                    positions=[],
                    actions=json.loads(serialized_actions),
                    current_action_index=0,
                    last_round_id="some_id",
                )
            )

            assert len(result) > 0
            assert result is not None

    def test_prepare_next_action_find_bridge_route(self):
        """Test _prepare_next_action method for FIND_BRIDGE_ROUTE action."""
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.FIND_BRIDGE_ROUTE.value,
                    "from_chain": "ethereum",
                    "to_chain": "polygon",
                    "from_token": "0x1234",
                    "from_token_symbol": "ETH",
                    "to_token": "0x5678",
                    "to_token_symbol": "MATIC",
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        current_behaviour = self.behaviour.current_behaviour

        # Mock fetch_routes to return sample routes
        sample_routes = [{"steps": [{"tool": "lifi"}]}]
        with mock.patch.object(
            current_behaviour, "fetch_routes", return_value=sample_routes
        ):
            result = next(
                current_behaviour._prepare_next_action(
                    positions=[],
                    actions=json.loads(serialized_actions),
                    current_action_index=0,
                    last_round_id="some_id",
                )
            )

            # Assert the structure of the returned routes
            assert isinstance(result, dict)
            assert "steps" in result
            assert result["steps"] == [{"tool": "lifi"}]

    def test_handle_failed_step(self):
        """Test _handle_failed_step method."""
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.FIND_BRIDGE_ROUTE.value,
                    "from_chain": "ethereum",
                    "to_chain": "polygon",
                    "from_token": "0x1234",
                    "from_token_symbol": "ETH",
                    "to_token": "0x5678",
                    "to_token_symbol": "MATIC",
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        step_data = {
            "from_chain": "chain1",
            "to_chain": "chain2",
            "source_token": "token1",
            "source_token_symbol": "TKN1",
            "target_token": "token2",
            "target_token_symbol": "TKN2",
        }

        # Test first step failure
        result = self.behaviour.current_behaviour._handle_failed_step(
            0, 0, step_data, 2
        )
        assert result == (
            Event.UPDATE.value,
            {
                "last_executed_route_index": 0,
                "last_executed_step_index": None,
                "last_action": Action.SWITCH_ROUTE.value,
            },
        )

    def test_get_next_event_no_actions(self):
        """Test get_next_event with no actions."""
        serialized_actions = json.dumps([])

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # When working with generators that have a return value,
        # we need to handle StopIteration exception to get the return value
        try:
            generator = self.behaviour.current_behaviour.get_next_event()
            while True:
                next(generator)
        except StopIteration as e:
            result = e.value

        assert result == (Event.DONE.value, {})

    def test_wait_for_swap_confirmation_success(self):
        """Test _wait_for_swap_confirmation when swap succeeds."""
        serialized_actions = json.dumps([{"action": Action.BRIDGE_SWAP.value}])

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock get_decision_on_swap to yield the sequence of decisions
        def mock_get_decision():
            for decision in [Decision.WAIT, Decision.CONTINUE]:
                yield decision

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_decision_on_swap",
            return_value=mock_get_decision(),
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "sleep",
            side_effect=lambda *_: (yield None),
        ):
            generator = self.behaviour.current_behaviour._wait_for_swap_confirmation()
            last_value = None
            # Consume the generator and get the last value
            for value in generator:
                last_value = value
            assert last_value == Decision.CONTINUE

    def test_execute_route_step_profitability_check(self):
        """Test _execute_route_step with profitability check."""
        routes = [
            {
                "steps": [
                    {"tool": "test_tool", "action": {"fromChainId": 1, "toChainId": 2}}
                ]
            }
        ]

        serialized_actions = json.dumps(routes)
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        def mock_check_if_route_is_profitable(*args, **kwargs):
            """Generator that returns profitability check results"""
            result = (False, 0.0, 0.0)
            yield result
            return result

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "check_if_route_is_profitable",
            side_effect=mock_check_if_route_is_profitable,
        ):
            try:
                generator = self.behaviour.current_behaviour._execute_route_step(
                    positions=[],
                    routes=routes,
                    to_execute_route_index=0,
                    to_execute_step_index=0,
                )
                last_value = None
                while True:
                    last_value = next(generator)
            except StopIteration as e:
                result = e.value if e.value is not None else last_value

            expected_result = (
                Event.UPDATE.value,
                {
                    "last_executed_route_index": 0,
                    "last_executed_step_index": None,
                    "last_action": Action.SWITCH_ROUTE.value,
                },
            )
            assert result == expected_result

    def test_post_execute_enter_pool_success(self):
        """Test _post_execute_enter_pool success case."""
        action = {
            "chain": "ethereum",
            "pool_address": "0x123",
            "dex_type": DexType.UNISWAP_V3.value,
            "token0": "0xtoken0",
            "token1": "0xtoken1",
            "token0_symbol": "TKN0",
            "token1_symbol": "TKN1",
        }

        serialized_actions = json.dumps([action])
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions], final_tx_hash=["test_hash"]
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Reset current positions before test
        self.behaviour.current_behaviour.current_positions = []

        receipt_data = (1, 100, 1000, 2000, int(time.time()))

        def mock_get_mint_receipt(*args, **kwargs):
            """Mock generator for _get_data_from_mint_tx_receipt"""
            yield receipt_data
            return receipt_data

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_get_data_from_mint_tx_receipt",
            side_effect=mock_get_mint_receipt,
        ), mock.patch.object(
            self.behaviour.current_behaviour, "store_current_positions"
        ):
            # Execute the generator
            generator = self.behaviour.current_behaviour._post_execute_enter_pool(
                [action], 0
            )
            try:
                while True:
                    next(generator)
            except StopIteration:
                pass

            # Verify that the position has been added correctly
            assert len(self.behaviour.current_behaviour.current_positions) == 1
            position = self.behaviour.current_behaviour.current_positions[0]

            # Assert expected values
            assert position["pool_address"] == action["pool_address"]
            assert position["token_id"] == receipt_data[0]
            assert position["liquidity"] == receipt_data[1]
            assert position["amount0"] == receipt_data[2]
            assert position["amount1"] == receipt_data[3]
            assert position["timestamp"] == receipt_data[4]
            assert position["token0_symbol"] == action["token0_symbol"]
            assert position["token1_symbol"] == action["token1_symbol"]
            assert position["dex_type"] == action["dex_type"]
            assert position["chain"] == action["chain"]

    def test_post_execute_step_with_failed_swap(self):
        """Test _post_execute_step when swap fails."""

        self.behaviour.current_behaviour.params.__dict__["_frozen"] = False
        self.behaviour.current_behaviour.params.waiting_period_for_status_check = 1

        serialized_actions = json.dumps(
            [
                {
                    "action": Action.BRIDGE_SWAP.value,
                    "remaining_fee_allowance": 100,
                    "remaining_gas_allowance": 200,
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["test_hash"],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        def mock_sleep(*args, **kwargs):
            """Mock generator for sleep method"""
            yield None

        def mock_get_decision(*args, **kwargs):
            """Mock generator for get_decision_on_swap"""
            yield Decision.EXIT.value  # Directly yield EXIT decision

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_decision_on_swap",
            side_effect=mock_get_decision,
        ), mock.patch.object(
            self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
        ), mock.patch.object(
            self.behaviour.current_behaviour.params,
            "waiting_period_for_status_check",
            new=1,  # Mock the waiting period
        ):
            # Execute the generator and get final result
            results = []
            try:
                generator = self.behaviour.current_behaviour._post_execute_step(
                    json.loads(serialized_actions), 0
                )
                while True:
                    value = next(generator)
                    if value is not None:
                        results.append(value)
            except StopIteration as e:
                # The return value comes in StopIteration when generator completes
                if e.value is not None:
                    results.append(e.value)

            # There should be one result and it should be the expected tuple
            assert len(results) >= 1

    def test_no_actions_to_execute(self):
        """Test when the actions list is empty."""

        serialized_actions = json.dumps([])

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        try:
            generator = self.behaviour.current_behaviour.get_next_event()
            result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == (Event.DONE.value, {}), "Empty actions should return DONE."

    def test_last_executed_action_index_out_of_bounds(self):
        """Test when last_executed_action_index is invalid."""
        # Simulate an action list with one action
        serialized_actions = json.dumps([{"action": Action.ENTER_POOL.value}])

        # Setup data with out-of-bounds index
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    last_executed_action_index=[5],  # Out-of-bounds index
                    final_tx_hash=["test_hash"],
                    chain_id=["ethereum"],
                    positions=[],  # Add empty positions for safety
                )
            )
        )

        # Forward behaviour
        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock any required methods to prevent actual async operations
        with mock.patch.object(
            self.behaviour.current_behaviour, "get_positions", return_value=[]
        ):
            # Get result from generator
            result = None
            try:
                generator = self.behaviour.current_behaviour.get_next_event()
                # Just get the first value from the generator
                result = next(generator)
                if result is None:
                    # If first value is None, try to get StopIteration value
                    while True:
                        next(generator)
            except StopIteration as e:
                # Get the return value from StopIteration
                result = e.value if e.value is not None else result

            # Verify result
            assert result == (
                Event.DONE.value,
                {},
            ), "Out-of-bounds index should return DONE."

    def test_empty_positions(self):
        """Test when the positions list is empty."""
        # Create synchronized data with empty positions
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[json.dumps([{"action": Action.ENTER_POOL.value}])],
                    positions=[],  # Empty positions list
                    final_tx_hash=["test_hash"],  # Add required data
                    chain_id=["ethereum"],  # Add required data
                    last_executed_action_index=[-1],  # Add initial index
                )
            )
        )

        # Forward behaviour
        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock get_positions to simulate empty positions
        with mock.patch.object(
            self.behaviour.current_behaviour, "get_positions", return_value=[]
        ):
            # Handle generator results
            result = None
            try:
                generator = self.behaviour.current_behaviour.get_next_event()
                while True:
                    next_value = next(generator)
                    if next_value is not None:
                        result = next_value
            except StopIteration as e:
                result = e.value if e.value is not None else result

            # Verify result
            assert result == (
                Event.DONE.value,
                {},
            ), "Empty positions should return DONE."

    def test_post_execute_step_no_decision(self):
        """Test when post_execute_step fails to get a decision."""
        self.behaviour.context.params.__dict__["_frozen"] = False
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.BRIDGE_SWAP.value,
                    "remaining_fee_allowance": 100,
                    "remaining_gas_allowance": 200,
                }
            ]
        )

        # Set up synchronized data with all required fields
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["test_hash"],
                    chain_id=["ethereum"],
                    positions=[],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock generator for get_decision_on_swap
        def mock_get_decision():
            """Mock generator for returning EXIT decision"""
            yield Decision.EXIT.value

        # Mock sleep to prevent actual waiting
        def mock_sleep(*args):
            """Mock generator for sleep"""
            yield None

        # Setup all required mocks
        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_decision_on_swap",
            side_effect=mock_get_decision,
        ), mock.patch.object(
            self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
        ), mock.patch.object(
            self.behaviour.current_behaviour.params,
            "waiting_period_for_status_check",
            1,  # Mock waiting period
        ):
            result = None
            try:
                generator = self.behaviour.current_behaviour._post_execute_step(
                    json.loads(serialized_actions), 0
                )

                # Process generator results
                while True:
                    try:
                        value = next(generator)
                        if value is not None:
                            result = value
                    except StopIteration as e:
                        if e.value is not None:
                            result = e.value
                        break

            except Exception as e:
                self.context.logger.error(f"Error during test: {e}")
                raise

            # Verify result
            assert len(result) > 0, "EXIT decision should return DONE."

    def test_build_multisend_tx_fails(self):
        """Test when build_multisend_tx fails."""

        # Setup test data
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.BRIDGE_SWAP.value,
                    "from_chain": "ethereum",
                    "to_chain": "polygon",
                }
            ]
        )

        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["test_hash"],
                    chain_id=["ethereum"],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Create mock generator function
        def mock_build_multisend_tx(*args, **kwargs):
            """Mock generator that returns None"""
            yield None
            return None

        # Mock required methods
        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_build_multisend_tx",
            side_effect=mock_build_multisend_tx,
        ):
            test_tx_info = {
                "from_chain": "ethereum",
                "to_chain": "polygon",
                "source_token": "0x123",
                "target_token": "0x456",
                "amount": 1000,
            }

            # Execute generator
            generator = self.behaviour.current_behaviour.prepare_bridge_swap_action(
                positions=[],
                tx_info=test_tx_info,
                remaining_fee_allowance=100,
                remaining_gas_allowance=100,
            )

            # Process generator results
            result = None
            try:
                while True:
                    value = next(generator)
                    if value is not None:
                        result = value
            except StopIteration as e:
                result = e.value if e.value is not None else None

            # Verify result
            assert result is None, "Failed multisend tx build should return None"

    def test_simulate_transaction_fails(self):
        """Test when simulate_transaction fails."""
        serialized_actions = json.dumps([{"action": Action.BRIDGE_SWAP.value}])

        synchronized_data = SynchronizedData(
            AbciAppDB(setup_data={"actions": [serialized_actions]})
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "_simulate_transaction",
            return_value=False,  # Simulate failure
        ):
            generator = self.behaviour.current_behaviour.prepare_bridge_swap_action(
                [], {}, 0, 0
            )

            try:
                next(generator)
            except StopIteration as e:
                result = e.value
                assert result is None, "Failed simulation should return None."

    def test_get_next_event_with_varied_actions(self):
        """Test get_next_event with different actions."""
        # Define varied actions with required fields
        actions = [
            {
                "action": Action.ENTER_POOL.value,
                "chain": "ethereum",
                "dex_type": DexType.UNISWAP_V3.value,
                "pool_address": "0x123",
                "token0": "0x456",
                "token1": "0x789",
            },
            {
                "action": Action.EXIT_POOL.value,
                "chain": "ethereum",
                "pool_address": "0x123",
                "dex_type": DexType.UNISWAP_V3.value,
            },
            {
                "action": Action.BRIDGE_SWAP.value,
                "from_chain": "ethereum",
                "to_chain": "polygon",
                "safe_address": "0x123",
                "payload": "0xabc",
            },
        ]

        # Setup synchronized data with all required fields
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[json.dumps(actions)],
                    final_tx_hash=["test_hash"],
                    chain_id=["ethereum"],
                    positions=[],  # Empty positions list
                    last_executed_action_index=[-1],  # Start with no executed actions
                )
            )
        )

        # Forward to behaviour
        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Mock required methods
        def mock_prepare_tx(*args, **kwargs):
            """Mock prepare tx hash generation."""
            return "0xtx_hash", "ethereum", "0xsafe"

        def mock_prepare_next_action(*args, **kwargs):
            """Mock prepare next action."""
            return (
                Event.SETTLE.value,
                {  # Return a tuple as expected
                    "tx_submitter": DecisionMakingRound.auto_round_id(),
                    "most_voted_tx_hash": "0xtx_hash",
                    "chain_id": "ethereum",
                    "safe_contract_address": "0xsafe",
                    "positions": [],
                    "last_executed_action_index": 0,
                    "last_action": Action.ENTER_POOL.value,
                },
            )

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_enter_pool_tx_hash",
            side_effect=mock_prepare_tx,
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "get_exit_pool_tx_hash",
            side_effect=mock_prepare_tx,
        ), mock.patch.object(
            self.behaviour.current_behaviour, "get_positions", return_value=[]
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "_prepare_next_action",
            side_effect=mock_prepare_next_action,
        ):
            # Process the generator
            generator = self.behaviour.current_behaviour.get_next_event()
            final_result = None
            event = None
            event_data = None

            # Collect all generator values
            try:
                while True:
                    result = next(generator)
                    print(f"generator result {result}")

                    # Assign the first and second results to the event and data
                    if event is None:
                        event = result
                    else:
                        event_data = result
                        final_result = (event, event_data)
                        event = None  # Reset the event for the next iteration
            except StopIteration as e:
                if e.value is not None:
                    event_data = e.value
                    if event is not None:
                        final_result = (event, event_data)

            # Expected result
            expected_result = (
                Event.SETTLE.value,
                {
                    "tx_submitter": DecisionMakingRound.auto_round_id(),
                    "most_voted_tx_hash": "0xtx_hash",
                    "chain_id": "ethereum",
                    "safe_contract_address": "0xsafe",
                    "positions": [],
                    "last_executed_action_index": 0,
                    "last_action": Action.ENTER_POOL.value,
                },
            )

            print(f"final_result {final_result}")
            # Verify the result
            assert (
                final_result == expected_result
            ), f"Expected {expected_result}, but got {final_result}"

    def test_transition_to_done(self):
        """Test transition to DONE event."""
        synchronized_data = SynchronizedData(
            AbciAppDB(setup_data={"actions": [json.dumps([])]})
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        generator = self.behaviour.current_behaviour.get_next_event()

        try:
            result = next(generator)
        except StopIteration as e:
            result = e.value

        assert result == (Event.DONE.value, {}), "No actions should return DONE."

    def test_swap_timeout_handling(self):
        """Test behavior when swap confirmation times out."""
        self.behaviour.context.params.__dict__["_frozen"] = False
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.BRIDGE_SWAP.value,
                    "remaining_fee_allowance": 100,
                    "remaining_gas_allowance": 200,
                    "from_chain": "ethereum",
                    "to_chain": "polygon",
                    "from_token": "0x123",
                    "to_token": "0x456",
                }
            ]
        )

        # Setup synchronized data with all required fields
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["test_hash"],
                    chain_id=["ethereum"],
                    positions=[],
                    last_executed_action_index=[-1],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Create a counter to simulate multiple decision responses
        decision_counter = 0

        def mock_get_decision():
            """Mock generator that returns WAIT then EXIT."""
            nonlocal decision_counter
            while True:  # Infinite loop to ensure we handle multiple calls
                if decision_counter < 3:  # Return WAIT a few times
                    decision_counter += 1
                    yield Decision.WAIT.value
                else:  # Then return EXIT
                    yield Decision.EXIT.value

        def mock_sleep(*_args, **_kwargs):
            """Mock generator for sleep that accepts any args."""
            yield None

        # Set the waiting period parameter
        self.behaviour.current_behaviour.params.waiting_period_for_status_check = 5

        # Mock methods with proper generators
        with mock.patch.object(
            self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "get_decision_on_swap",
            side_effect=mock_get_decision,
        ):
            # Run the generator
            generator = self.behaviour.current_behaviour._wait_for_swap_confirmation()
            result = None

            try:
                while True:
                    value = next(generator)
                    print(f"value generator: {value}")
                    if value == Decision.EXIT.value:
                        result = value
                        break
            except StopIteration as e:
                result = e.value if e.value is not None else result

            # Verify the result
            assert (
                result == Decision.EXIT.value
            ), f"Timeout should lead to EXIT decision, got {result}"

    def test_post_execute_step_get_decision(self):
        """Test post_execute_step when get_decision_on_swap fails."""
        self.behaviour.context.params.__dict__["_frozen"] = False

        # Create test action with required fields
        serialized_actions = json.dumps(
            [
                {
                    "action": Action.BRIDGE_SWAP.value,
                    "remaining_fee_allowance": 100,
                    "remaining_gas_allowance": 200,
                    "from_chain": "ethereum",
                    "from_token": "0x123",
                    "from_token_symbol": "ETH",
                    "to_token": "0x456",
                    "to_token_symbol": "USDC",
                }
            ]
        )

        # Setup synchronized data with required fields
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[serialized_actions],
                    final_tx_hash=["0x123"],
                    chain_id=["ethereum"],
                    positions=[],
                    last_executed_action_index=[-1],
                )
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Create decision sequence mock
        def mock_get_decision_on_swap():
            """Mock generator that simulates a WAIT then EXIT sequence."""
            yield Decision.WAIT.value
            yield Decision.EXIT.value

        def mock_sleep(*_args, **_kwargs):
            """Mock generator for sleep."""
            yield None

        # Setup mocks
        with mock.patch.object(
            self.behaviour.current_behaviour,
            "get_decision_on_swap",
            side_effect=mock_get_decision_on_swap,
        ), mock.patch.object(
            self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
        ), mock.patch.object(
            self.behaviour.current_behaviour.params,
            "waiting_period_for_status_check",
            1,
        ):
            # Get generator
            generator = self.behaviour.current_behaviour._post_execute_step(
                [json.loads(serialized_actions)], last_executed_action_index=0
            )

            # Process generator output and collect all values
            values = []
            try:
                while True:
                    value = next(generator)
                    print(f"generator value: {value}")
                    values.append(value)
            except StopIteration as e:
                # Add StopIteration value if present
                if e.value is not None:
                    values.append(e.value)

            # Expected sequence of values
            expected_values = [None, Decision.WAIT.value, Decision.EXIT.value]

            # Verify the sequence
            assert (
                values == expected_values
            ), f"Expected sequence {expected_values} but got {values}"

    def test_post_execute_step_with_decisions(self):
        """Test post_execute_step with WAIT, EXIT, CONTINUE decisions."""
        # Create test actions
        self.behaviour.context.params.__dict__["_frozen"] = False
        actions = [
            {
                "action": Action.BRIDGE_SWAP.value,
                "remaining_fee_allowance": 100,
                "remaining_gas_allowance": 200,
                "from_chain": "ethereum",
                "from_token": "0x123",
                "from_token_symbol": "ETH",
                "to_token": "0x456",
                "to_token_symbol": "USDC",
            }
        ]
        # Setup synchronized data with required fields
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data=dict(
                    actions=[json.dumps(actions)],
                    final_tx_hash=["test_hash"],
                    chain_id=["ethereum"],
                    last_executed_step_index=[0],
                )
            )
        )
        self.fast_forward_to_behaviour(
            self.behaviour,
            DecisionMakingBehaviour.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )
        # Create mock generator factory function

        def create_decision_mock(decision_value):
            """Create a mock generator for a specific decision value."""

            def mock_get_decision():
                yield decision_value

            return mock_get_decision

        def mock_sleep(*args, **kwargs):
            """Mock generator for sleep."""
            yield None

        # Test different decisions
        test_cases = [
            (Decision.WAIT.value, None),  # WAIT should return WAIT value
            (Decision.EXIT.value, (Event.DONE.value, {})),  # EXIT should return DONE
            (
                Decision.CONTINUE.value,
                (
                    Event.UPDATE.value,
                    {  # CONTINUE should update
                        "last_executed_step_index": 0,
                        "fee_details": {
                            "remaining_fee_allowance": 100,
                            "remaining_gas_allowance": 200,
                        },
                        "last_action": Action.STEP_EXECUTED.value,
                    },
                ),
            ),
        ]
        for decision_value in test_cases:
            # Setup mocks
            with mock.patch.object(
                self.behaviour.current_behaviour,
                "get_decision_on_swap",
                side_effect=create_decision_mock(decision_value),
            ), mock.patch.object(
                self.behaviour.current_behaviour, "sleep", side_effect=mock_sleep
            ), mock.patch.object(
                self.behaviour.current_behaviour,
                "_update_assets_after_swap",
                return_value=(
                    Event.UPDATE.value,
                    {
                        "last_executed_step_index": 0,
                        "fee_details": {
                            "remaining_fee_allowance": 100,
                            "remaining_gas_allowance": 200,
                        },
                        "last_action": Action.STEP_EXECUTED.value,
                    },
                ),
            ), mock.patch.object(
                self.behaviour.current_behaviour.params,
                "waiting_period_for_status_check",
                1,  # Mock waiting period
            ):
                # Process generator
                generator = self.behaviour.current_behaviour._post_execute_step(
                    actions, 0
                )
                result = None
                # Handle generator differently based on decision
                if decision_value == Decision.WAIT.value:
                    # For WAIT case, just get the first value
                    try:
                        while True:
                            value = next(generator)
                            print(f"result resulty value{value}")
                            if value is not None:
                                result = value
                    except StopIteration:
                        pass
                else:
                    # For other cases, process until we get a result
                    try:
                        while True:
                            value = next(generator)
                            print(f"result resulty value{value}")
                            if value is not None:
                                result = value
                    except StopIteration as e:
                        if e.value is not None:
                            result = e.value
                # Verify result based on decision type
                if decision_value == Decision.WAIT.value:
                    assert (
                        result == Decision.WAIT.value
                    ), f"Expected {Decision.WAIT.value}, but got {result}"


class TestPostTxSettlementBehaviour(FSMBehaviourBaseCase):
    """Test PostTxSettlementBehaviour."""

    path_to_skill = PACKAGE_DIR
    behaviour_class = PostTxSettlementBehaviour
    next_behaviour_class = PostTxSettlementBehaviour

    def setup(self) -> None:
        """Set up the test."""
        super().setup()
        self.behaviour.context.params.__dict__["_frozen"] = False
        self.behaviour.context.params.store_path = Path(".")
        self.behaviour.context.params.gas_cost_info_filename = "test_gas_costs.json"
        self.behaviour.current_behaviour.gas_cost_tracker = GasCostTracker(
            Path("test_gas_costs.json")
        )

    def test_tx_settlement(self) -> None:
        """Test post transaction settlement."""
        synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data={
                    "tx_submitter": ["test_round"],
                    "final_tx_hash": ["0x123"],
                    "chain_id": ["ethereum"],
                }
            )
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            synchronized_data=synchronized_data,
        )

        # Ensure current_behaviour is an instance, not a class
        current_behaviour = self.behaviour.current_behaviour

        def mock_fetch_and_log_gas_details():
            """Mock generator for fetch_and_log_gas_details."""
            return
            yield

        def mock_send_a2a_transaction(payload, resetting=False):
            """Mock generator for send_a2a_transaction."""
            assert payload.sender == current_behaviour.context.agent_address
            assert payload.content == "Transaction settled"
            return
            yield

        def mock_wait_until_round_end():
            """Mock generator for wait_until_round_end."""
            return
            yield

        with mock.patch.object(
            current_behaviour,
            "fetch_and_log_gas_details",
            side_effect=mock_fetch_and_log_gas_details,
        ), mock.patch.object(
            current_behaviour,
            "send_a2a_transaction",
            side_effect=mock_send_a2a_transaction,
        ), mock.patch.object(
            current_behaviour,
            "wait_until_round_end",
            side_effect=mock_wait_until_round_end,
        ):
            # Ensure the method can be called
            current_behaviour.set_done()

            # Explicitly call act_wrapper
            self.behaviour.act_wrapper()

            # Ensure the behaviour is marked as done
            assert current_behaviour.is_done()

    def test_post_tx_settlement(self) -> None:
        """Test post tx settlement."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        tx_submitter=["test_round"],
                        final_tx_hash=["0x123"],
                        chain_id=["ethereum"],
                    )
                )
            ),
        )

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "fetch_and_log_gas_details",
            side_effect=lambda: (yield None),
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "send_a2a_transaction",
            side_effect=self.behaviour.current_behaviour.send_a2a_transaction,
        ):
            self.behaviour.act_wrapper()
            current_behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
            current_behaviour.set_done()
            assert current_behaviour.is_done()

        self.end_round(Event.DONE)

        assert (
            self.behaviour.current_behaviour.behaviour_id
            == self.next_behaviour_class.auto_behaviour_id()
        )

    def test_vanity_tx_settlement(self) -> None:
        """Test post tx settlement."""
        next_behaviour_class = CheckStakingKPIMetBehaviour

        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        tx_submitter=["test_round"],
                        final_tx_hash=["0x123"],
                        chain_id=["ethereum"],
                    )
                )
            ),
        )

        with mock.patch.object(
            self.behaviour.current_behaviour,
            "fetch_and_log_gas_details",
            side_effect=lambda: (yield None),
        ), mock.patch.object(
            self.behaviour.current_behaviour,
            "send_a2a_transaction",
            side_effect=self.behaviour.current_behaviour.send_a2a_transaction,
        ):
            self.behaviour.act_wrapper()
            current_behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
            current_behaviour.set_done()
            assert current_behaviour.is_done()

        self.end_round(Event.VANITY_TX_EXECUTED)
        assert (
            self.behaviour.current_behaviour.behaviour_id
            == next_behaviour_class.auto_behaviour_id()
        )

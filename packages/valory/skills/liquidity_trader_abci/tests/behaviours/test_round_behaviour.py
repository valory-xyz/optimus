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

"""Test the behaviours/round_behaviour.py module of the liquidity_trader_abci skill."""

# pylint: skip-file

from packages.valory.skills.liquidity_trader_abci.behaviours.apr_population import (
    APRPopulationBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.call_checkpoint import (
    CallCheckpointBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.check_staking_kpi_met import (
    CheckStakingKPIMetBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.decision_making import (
    DecisionMakingBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.evaluate_strategy import (
    EvaluateStrategyBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.fetch_strategies import (
    FetchStrategiesBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.get_positions import (
    GetPositionsBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.post_tx_settlement import (
    PostTxSettlementBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.round_behaviour import (
    LiquidityTraderRoundBehaviour,
)
from packages.valory.skills.liquidity_trader_abci.behaviours.withdraw_funds import (
    WithdrawFundsBehaviour,
)


def test_import() -> None:
    """Test that the round_behaviour module can be imported."""
    import packages.valory.skills.liquidity_trader_abci.behaviours.round_behaviour  # noqa


class TestLiquidityTraderRoundBehaviour:
    """Test LiquidityTraderRoundBehaviour class."""

    def test_behaviours_set(self) -> None:
        """Test behaviours contains all expected behaviour classes."""
        expected_behaviours = {
            CallCheckpointBehaviour,
            CheckStakingKPIMetBehaviour,
            GetPositionsBehaviour,
            APRPopulationBehaviour,
            EvaluateStrategyBehaviour,
            DecisionMakingBehaviour,
            PostTxSettlementBehaviour,
            FetchStrategiesBehaviour,
            WithdrawFundsBehaviour,
        }
        actual = set(LiquidityTraderRoundBehaviour.behaviours)
        assert actual == expected_behaviours

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
)

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
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
from packages.valory.skills.liquidity_trader_abci.rounds import LiquidityTraderAbciApp


class LiquidityTraderRoundBehaviour(AbstractRoundBehaviour):
    """LiquidityTraderRoundBehaviour"""

    initial_behaviour_cls = CallCheckpointBehaviour
    abci_app_cls = LiquidityTraderAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [
        CallCheckpointBehaviour,
        CheckStakingKPIMetBehaviour,
        GetPositionsBehaviour,
        APRPopulationBehaviour,
        EvaluateStrategyBehaviour,
        DecisionMakingBehaviour,
        PostTxSettlementBehaviour,
        FetchStrategiesBehaviour,
    ]

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

"""This module contains the behaviours for the 'trader_decision_maker_abci' skill."""

import json
from abc import ABC
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviour_utils import BaseBehaviour
from packages.valory.skills.abstract_round_abci.behaviours import AbstractRoundBehaviour
from packages.valory.skills.abstract_round_abci.common import (
    RandomnessBehaviour as RandomnessBehaviourBase,
)
from packages.valory.skills.trader_decision_maker_abci.models import Params
from packages.valory.skills.trader_decision_maker_abci.payloads import (
    RandomnessPayload,
    TraderDecisionMakerPayload,
)
from packages.valory.skills.trader_decision_maker_abci.policy import EGreedyPolicy
from packages.valory.skills.trader_decision_maker_abci.rounds import (
    Position,
    RandomnessRound,
    SynchronizedData,
    TraderDecisionMakerAbciApp,
    TraderDecisionMakerRound,
)


DeserializedType = TypeVar("DeserializedType")


POLICY_STORE = "policy_store.json"
POSITIONS_STORE = "positions.json"
STRATEGIES_STORE = "strategies.json"


class RandomnessBehaviour(RandomnessBehaviourBase):
    """Retrieve randomness."""

    matching_round = RandomnessRound
    payload_class = RandomnessPayload


class TraderDecisionMakerBehaviour(BaseBehaviour, ABC):
    """A behaviour in which the agents select a trading strategy."""

    matching_round: Type[AbstractRound] = TraderDecisionMakerRound

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Behaviour."""
        super().__init__(**kwargs)
        base_dir = Path(self.context.data_dir)
        self.policy_path = base_dir / POLICY_STORE
        self.positions_path = base_dir / POSITIONS_STORE
        self.strategies_path = base_dir / STRATEGIES_STORE
        self.strategies: Tuple[str, ...] = tuple(self.context.shared_state.keys())

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return cast(Params, self.context.params)

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return SynchronizedData(super().synchronized_data.db)

    @property
    def policy(self) -> EGreedyPolicy:
        """Get the policy."""
        if self._policy is None:
            raise ValueError(
                "Attempting to retrieve the policy before it has been established."
            )
        return self._policy

    @property
    def is_first_period(self) -> bool:
        """Return whether it is the first period of the service."""
        return self.synchronized_data.period_count == 0

    @property
    def positions(self) -> List[Position]:
        """Get the positions of the service."""
        if self.is_first_period:
            positions = self._try_recover_from_store(
                self.positions_path,
                Position.from_json,
            )
            if positions is not None:
                return positions
            return []
        return self.synchronized_data.positions

    def _adjust_policy_strategies(self, local: List[str]) -> None:
        """Add or remove strategies from the locally stored policy to match the strategies given via the config."""
        # remove strategies if they are not available anymore
        # process the indices in a reverse order to avoid index shifting when removing the unavailable strategies later
        reversed_idx = range(len(local) - 1, -1, -1)
        removed_idx = [idx for idx in reversed_idx if local[idx] not in self.strategies]
        self.policy.remove_strategies(removed_idx)

        # add strategies if there are new ones available
        # process the indices in a reverse order to avoid index shifting when adding the new strategies later
        reversed_idx = range(len(self.strategies) - 1, -1, -1)
        new_idx = [idx for idx in reversed_idx if self.strategies[idx] not in local]
        self.policy.add_new_strategies(new_idx)

    def _set_policy(self) -> None:
        """Set the E Greedy Policy."""
        if not self.is_first_period:
            self._policy = self.synchronized_data.policy
            return

        self._policy = self._get_init_policy()
        local_strategies = self._try_recover_from_store(self.strategies_path)
        if local_strategies is not None:
            self._adjust_policy_strategies(local_strategies)

    def _get_init_policy(self) -> EGreedyPolicy:
        """Get the initial policy"""
        # try to read the policy from the policy store
        policy = self._try_recover_from_store(
            self.policy_path, lambda policy_: EGreedyPolicy(**policy_)
        )
        if policy is not None:
            # we successfully recovered the policy, so we return it
            return policy

        # we could not recover the policy, so we create a new one
        n_relevant = len(self.strategies)
        policy = EGreedyPolicy.initial_state(self.params.epsilon, n_relevant)
        return policy

    def _try_recover_from_store(
        self,
        path: Path,
        deserializer: Optional[Callable[[Any], DeserializedType]] = None,
    ) -> Optional[DeserializedType]:
        """Try to recover a previously saved file from the policy store."""
        try:
            with open(path, "r") as stream:
                res = json.load(stream)
                if deserializer is None:
                    return res
                return deserializer(res)
        except Exception as e:
            self.context.logger.warning(
                f"Could not recover file from the policy store: {e}."
            )
            return None

    def select_strategy(self) -> Optional[str]:
        """Select a strategy based on an e-greedy policy and return its index."""
        self._set_policy()
        randomness = self.synchronized_data.most_voted_randomness
        selected_idx = self.policy.select_strategy(randomness)
        selected = self.strategies[selected_idx] if selected_idx is not None else "NaN"
        self.context.logger.info(f"Selected strategy: {selected}.")
        return selected

    def _store_policy(self) -> None:
        """Store the policy."""
        with open(self.policy_path, "w") as policy_stream:
            policy_stream.write(self.policy.serialize())

    def _store_available_strategies(self) -> None:
        """Store the available strategies."""
        with open(self.strategies_path, "w") as strategies_stream:
            json.dump(self.strategies, strategies_stream)

    def async_act(self) -> Generator:
        """Do the action."""
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            policy = positions = None
            selected_strategy = self.select_strategy()
            if selected_strategy is not None:
                policy = self.policy.serialize()
                positions = json.dumps(self.positions, sort_keys=True)
                self._store_policy()
                self._store_available_strategies()

            payload = TraderDecisionMakerPayload(
                self.context.agent_address,
                policy,
                positions,
                selected_strategy,
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()


class TraderDecisionMakerRoundBehaviour(AbstractRoundBehaviour):
    """This behaviour manages the consensus stages for the TraderDecisionMakerBehaviour."""

    initial_behaviour_cls = RandomnessBehaviour
    abci_app_cls = TraderDecisionMakerAbciApp
    behaviours: Set[Type[BaseBehaviour]] = {
        RandomnessBehaviour,  # type: ignore
        TraderDecisionMakerBehaviour,  # type: ignore
    }

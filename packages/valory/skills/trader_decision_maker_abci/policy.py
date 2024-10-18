# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""This module contains an Epsilon Greedy Policy implementation."""

import json
import random
from dataclasses import dataclass
from typing import List, Optional, Union

from packages.valory.skills.trader_decision_maker_abci.utils import DataclassEncoder


RandomnessType = Union[int, float, str, bytes, bytearray, None]


def argmax(li: List) -> int:
    """Get the index of the max value within the provided list."""
    return li.index((max(li)))


@dataclass
class EGreedyPolicy:
    """An e-Greedy policy for the strategy selection."""

    eps: float
    counts: List[int]
    rewards: List[float]
    initial_value = 0

    @classmethod
    def initial_state(cls, eps: float, n_strategies: int) -> "EGreedyPolicy":
        """Return an instance on its initial state."""
        if n_strategies <= 0 or eps > 1 or eps < 0:
            raise ValueError(
                f"Cannot initialize an e Greedy Policy with {eps=} and {n_strategies=}"
            )

        return EGreedyPolicy(
            eps,
            [cls.initial_value] * n_strategies,
            [float(cls.initial_value)] * n_strategies,
        )

    @property
    def n_strategies(self) -> int:
        """Get the number of the policy's strategies."""
        return len(self.counts)

    @property
    def random_strategy(self) -> int:
        """Get the index of a strategy randomly."""
        return random.randrange(self.n_strategies)  # nosec

    @property
    def has_updated(self) -> bool:
        """Whether the policy has ever been updated since its genesis or not."""
        return sum(self.counts) > 0

    @property
    def reward_rates(self) -> List[float]:
        """Get the reward rates."""
        return [
            reward / count if count > 0 else 0
            for reward, count in zip(self.rewards, self.counts)
        ]

    @property
    def best_strategy(self) -> int:
        """Get the best strategy."""
        return argmax(self.reward_rates)

    def add_new_strategies(self, indexes: List[int], avoid_shift: bool = False) -> None:
        """Add new strategies to the current policy."""
        if avoid_shift:
            indexes = sorted(indexes, reverse=True)

        for i in indexes:
            self.counts.insert(i, self.initial_value)
            self.rewards.insert(i, float(self.initial_value))

    def remove_strategies(self, indexes: List[int], avoid_shift: bool = False) -> None:
        """Remove the knowledge for the strategies corresponding to the given indexes."""
        if avoid_shift:
            indexes = sorted(indexes, reverse=True)

        for i in indexes:
            try:
                del self.counts[i]
                del self.rewards[i]
            except IndexError as exc:
                error = "Attempted to remove strategies using incorrect indexes!"
                raise ValueError(error) from exc

    def select_strategy(self, randomness: RandomnessType) -> Optional[int]:
        """Select a strategy and return its index."""
        if self.n_strategies == 0:
            return None

        random.seed(randomness)
        if sum(self.reward_rates) == 0 or random.random() < self.eps:  # nosec
            return self.random_strategy

        return self.best_strategy

    def add_reward(self, index: int, reward: float = 0) -> None:
        """Add a reward for the strategy corresponding to the given index."""
        self.counts[index] += 1
        self.rewards[index] += reward

    def serialize(self) -> str:
        """Return the policy serialized."""
        return json.dumps(self, cls=DataclassEncoder, sort_keys=True)

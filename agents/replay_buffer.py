"""Replay buffer for off-policy DQN training."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int, observation_size: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        if observation_size <= 0:
            raise ValueError("observation_size must be positive")
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample a mini-batch of transitions."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.size:
            raise ValueError("cannot sample more transitions than currently stored")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size

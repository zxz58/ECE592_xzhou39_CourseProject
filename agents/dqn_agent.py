"""DQN agent for budget-constrained recalibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.networks import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """Minimal DQN implementation with target network."""

    def __init__(
        self,
        observation_size: int | None = None,
        action_size: int | None = None,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.98,
        replay_capacity: int = 20_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 8_000,
        target_update_interval: int = 200,
        min_replay_size: int = 500,
        seed: int = 0,
        device: str | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> None:
        if observation_size is None:
            observation_size = input_dim
        if action_size is None:
            action_size = output_dim
        if observation_size is None or action_size is None:
            raise ValueError("observation_size/action_size or input_dim/output_dim are required")
        if observation_size <= 0 or action_size <= 0:
            raise ValueError("observation_size and action_size must be positive")

        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.target_update_interval = int(target_update_interval)
        self.min_replay_size = int(min_replay_size)
        self.training_steps = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)

        self.q_network = QNetwork(observation_size, action_size, hidden_dim).to(self.device)
        self.target_network = QNetwork(observation_size, action_size, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_capacity, observation_size, seed=seed)

    def _epsilon(self) -> float:
        """Current linear-decay exploration rate."""
        fraction = min(self.training_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return float(self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start))

    def current_epsilon(self) -> float:
        """Expose the current exploration rate for logging."""
        return self._epsilon()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action using epsilon-greedy exploration during training."""
        epsilon = self._epsilon() if training else 0.0
        action = self.act(state, epsilon=epsilon)
        if training:
            self.training_steps += 1
        return action

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Compatibility wrapper for epsilon-greedy action selection."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_size))
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q_network(obs), dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition in replay memory."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> float | None:
        """Run one batch DQN update and periodically refresh the target network."""
        required_samples = max(self.min_replay_size, self.batch_size)
        if len(self.replay_buffer) < required_samples:
            return None
        loss = self.train_step(self.batch_size)
        if self.training_steps % max(self.target_update_interval, 1) == 0:
            self.update_target()
        return loss

    def train_step(self, batch_size: int | None = None) -> float | None:
        """Compatibility wrapper for one DQN update."""
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.replay_buffer) < batch_size:
            return None

        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(batch_size)
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rew = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_network(obs).gather(1, act).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(nxt).max(dim=1).values
            target = rew + self.gamma * (1.0 - done) * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        """Copy online network weights to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str | Path) -> None:
        """Save model weights and key dimensions."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.q_network.state_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("observation_size") != self.observation_size:
            raise ValueError(
                "checkpoint observation_size does not match current environment "
                f"({checkpoint.get('observation_size')} != {self.observation_size})"
            )
        if checkpoint.get("action_size") != self.action_size:
            raise ValueError(
                "checkpoint action_size does not match current environment "
                f"({checkpoint.get('action_size')} != {self.action_size})"
            )
        self.q_network.load_state_dict(checkpoint["state_dict"])
        self.target_network.load_state_dict(self.q_network.state_dict())

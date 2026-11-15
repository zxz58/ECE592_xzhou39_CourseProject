"""Neural network definitions."""

import torch
from torch import nn


class QNetwork(nn.Module):
    """Small MLP for binary recalibration decisions."""

    def __init__(self, observation_size: int, action_size: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

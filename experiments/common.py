"""Shared construction and persistence helpers for course experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from agents.dqn_agent import DQNAgent
from envs.recalibration_env import RecalibrationEnv
from envs.surrogate_model import DeterministicSEIRSurrogate
from envs.true_simulator import StochasticSEIRSimulator
from experiments.trajectories import BetaTrajectory


def make_env(config: dict, tier: str, trajectory: BetaTrajectory) -> RecalibrationEnv:
    """Build an isolated environment for one episode and beta trajectory."""
    simulator_config = deepcopy(config["simulator"])
    simulator_config.update(
        episode_length=len(trajectory.schedule),
        beta_schedule=trajectory.schedule.tolist(),
    )
    simulator = StochasticSEIRSimulator(**simulator_config)
    surrogate = DeterministicSEIRSurrogate(**config["surrogate"])
    env_config = config["environment"]
    fixed = tier == "fixed"
    horizon = len(trajectory.schedule)
    return RecalibrationEnv(
        true_simulator=simulator,
        surrogate_model=surrogate,
        termination_mode="fixed_horizon" if fixed else "epidemic",
        horizon=horizon,
        max_steps=horizon,
        extinction_threshold=env_config["extinction_threshold"],
        extinction_patience=env_config["extinction_patience"],
        min_steps_before_extinction_check=env_config["min_steps_before_extinction_check"],
        forecast_horizon=env_config["forecast_horizon"],
        recalibration_budget=env_config["recalibration_budget"],
        error_metric=env_config["error_metric"],
        state_features=env_config["state_features"],
        error_history_length=env_config["error_history_length"],
        immediate_error_weight=env_config["immediate_error_weight"],
        path_error_weight=env_config["path_error_weight"],
    )


def make_agent(config: dict, observation_size: int, seed: int) -> DQNAgent:
    """Create the one DQN architecture used by both tiers."""
    dqn = config["dqn"]
    return DQNAgent(
        observation_size=observation_size,
        action_size=2,
        hidden_dim=dqn["hidden_dim"],
        learning_rate=dqn["learning_rate"],
        gamma=dqn["gamma"],
        replay_capacity=dqn["replay_capacity"],
        batch_size=dqn["batch_size"],
        epsilon_start=dqn["epsilon_start"],
        epsilon_end=dqn["epsilon_end"],
        epsilon_decay_steps=dqn["epsilon_decay_steps"],
        target_update_interval=dqn["target_update_interval"],
        min_replay_size=dqn["min_replay_size"],
        seed=seed,
    )


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

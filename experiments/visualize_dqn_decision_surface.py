"""Visualize the learned DQN recalibration decision surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agents.dqn_agent import DQNAgent
from diagnostics.plots import plot_dqn_decision_surface
from experiments.common import make_env, save_csv
from utils.config import ensure_output_dirs, load_config
from utils.seed import set_seed


def _recent_errors(error_level: float, length: int, pattern: str) -> list[float]:
    """Create a recent-error vector with the requested trend shape."""
    if pattern == "flat_error":
        values = np.full(length, error_level, dtype=float)
    elif pattern == "increasing_error":
        values = np.linspace(0.25 * error_level, 1.75 * error_level, length)
    elif pattern == "decreasing_error":
        values = np.linspace(1.75 * error_level, 0.25 * error_level, length)
    else:
        raise ValueError(f"unknown error pattern: {pattern}")
    return np.clip(values, 0.0, None).tolist()


def _error_stats(errors: list[float]) -> tuple[float, float]:
    """Return rolling mean and the same simple trend used by the environment."""
    midpoint = max(len(errors) // 2, 1)
    earlier = float(np.mean(errors[:midpoint]))
    later = float(np.mean(errors[midpoint:]))
    return float(np.mean(errors)), later - earlier


def _build_observation(
    env,
    infectious_fraction: float,
    recent_errors: list[float],
    remaining_budget_fraction: float,
    days_since_recalibration_fraction: float,
    exposed_to_infectious_ratio: float,
    recovered_fraction: float,
) -> np.ndarray:
    """Build an observation vector using the environment's configured feature order."""
    i_fraction = float(infectious_fraction)
    e_fraction = float(exposed_to_infectious_ratio * i_fraction)
    r_fraction = float(recovered_fraction)
    if e_fraction + i_fraction + r_fraction >= 1.0:
        r_fraction = max(0.0, 1.0 - e_fraction - i_fraction)
    s_fraction = max(0.0, 1.0 - e_fraction - i_fraction - r_fraction)

    last_error = float(recent_errors[-1])
    rolling_mean, rolling_trend = _error_stats(recent_errors)
    values: list[float] = []
    for feature in env.state_features:
        if feature == "normalized_compartments":
            values.extend([s_fraction, e_fraction, i_fraction, r_fraction])
        elif feature == "recent_errors":
            values.extend(recent_errors)
        elif feature == "last_forecast_error":
            values.append(last_error)
        elif feature == "rolling_mean_forecast_error":
            values.append(rolling_mean)
        elif feature == "rolling_error_trend":
            values.append(rolling_trend)
        elif feature == "remaining_budget":
            values.append(float(remaining_budget_fraction))
        elif feature == "days_since_recalibration":
            values.append(float(days_since_recalibration_fraction))
    return np.asarray(values, dtype=np.float32)


def visualize(
    config_path: str,
    model_path: str,
    grid_size: int,
    max_infectious_fraction: float,
    max_error_level: float,
    remaining_budget_fraction: float,
    days_since_recalibration_fraction: float,
    exposed_to_infectious_ratio: float,
    recovered_fraction: float,
) -> None:
    """Compute DQN Q-values over a diagnostic state grid and save heatmaps."""
    config = load_config(config_path)
    ensure_output_dirs(config)
    set_seed(config["seed"])

    env = make_env(config)
    dqn_cfg = config["dqn"]
    agent = DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        hidden_dim=dqn_cfg["hidden_dim"],
        learning_rate=dqn_cfg["learning_rate"],
        gamma=dqn_cfg["gamma"],
        replay_capacity=dqn_cfg["replay_capacity"],
        batch_size=dqn_cfg["batch_size"],
        epsilon_start=dqn_cfg["epsilon_start"],
        epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay_steps=dqn_cfg["epsilon_decay_steps"],
        target_update_interval=dqn_cfg["target_update_interval"],
        min_replay_size=dqn_cfg["min_replay_size"],
        seed=config["seed"],
    )
    agent.load(model_path)
    agent.q_network.eval()

    infectious_values = np.linspace(0.0, max_infectious_fraction, grid_size)
    error_values = np.linspace(0.0, max_error_level, grid_size)
    patterns = ["decreasing_error", "flat_error", "increasing_error"]

    rows = []
    for pattern in patterns:
        observations = []
        keys = []
        for recent_error_level in error_values:
            errors = _recent_errors(
                error_level=float(recent_error_level),
                length=env.error_history_length,
                pattern=pattern,
            )
            rolling_mean, rolling_trend = _error_stats(errors)
            for infectious_fraction in infectious_values:
                observation = _build_observation(
                    env=env,
                    infectious_fraction=float(infectious_fraction),
                    recent_errors=errors,
                    remaining_budget_fraction=remaining_budget_fraction,
                    days_since_recalibration_fraction=days_since_recalibration_fraction,
                    exposed_to_infectious_ratio=exposed_to_infectious_ratio,
                    recovered_fraction=recovered_fraction,
                )
                observations.append(observation)
                keys.append(
                    {
                        "error_pattern": pattern,
                        "infectious_fraction": float(infectious_fraction),
                        "recent_error_level": float(recent_error_level),
                        "rolling_mean_forecast_error": rolling_mean,
                        "rolling_error_trend": rolling_trend,
                    }
                )

        with torch.no_grad():
            obs_tensor = torch.as_tensor(np.vstack(observations), dtype=torch.float32, device=agent.device)
            q_values = agent.q_network(obs_tensor).cpu().numpy()
            probabilities = torch.softmax(torch.as_tensor(q_values), dim=1).numpy()

        for key, q_value, probability in zip(keys, q_values, probabilities):
            rows.append(
                {
                    **key,
                    "remaining_budget_fraction": float(remaining_budget_fraction),
                    "days_since_recalibration_fraction": float(days_since_recalibration_fraction),
                    "q_wait": float(q_value[0]),
                    "q_recalibrate": float(q_value[1]),
                    "q_margin_recalibrate_minus_wait": float(q_value[1] - q_value[0]),
                    "softmax_p_wait": float(probability[0]),
                    "softmax_p_recalibrate": float(probability[1]),
                    "greedy_action": int(np.argmax(q_value)),
                }
            )

    results_dir = Path(config["outputs"]["results_dir"])
    figures_dir = Path(config["outputs"].get("figures_dir", results_dir / "figures"))
    surface = pd.DataFrame(rows)
    save_csv(surface, results_dir / "dqn_decision_surface_grid.csv")
    plot_dqn_decision_surface(surface, figures_dir)
    print(f"saved decision-surface grid: {results_dir / 'dqn_decision_surface_grid.csv'}")
    print(f"saved decision-surface heatmaps to: {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="results/dqn_model.pt")
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--max-infectious-fraction", type=float, default=0.08)
    parser.add_argument("--max-error-level", type=float, default=0.0015)
    parser.add_argument("--remaining-budget-fraction", type=float, default=0.5)
    parser.add_argument("--days-since-recalibration-fraction", type=float, default=14 / 365)
    parser.add_argument("--exposed-to-infectious-ratio", type=float, default=0.5)
    parser.add_argument("--recovered-fraction", type=float, default=0.10)
    args = parser.parse_args()
    visualize(
        config_path=args.config,
        model_path=args.model,
        grid_size=args.grid_size,
        max_infectious_fraction=args.max_infectious_fraction,
        max_error_level=args.max_error_level,
        remaining_budget_fraction=args.remaining_budget_fraction,
        days_since_recalibration_fraction=args.days_since_recalibration_fraction,
        exposed_to_infectious_ratio=args.exposed_to_infectious_ratio,
        recovered_fraction=args.recovered_fraction,
    )


if __name__ == "__main__":
    main()

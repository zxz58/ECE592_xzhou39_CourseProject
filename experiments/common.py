"""Shared experiment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from envs.recalibration_env import RecalibrationEnv
from envs.surrogate_model import DeterministicSEIRSurrogate
from envs.true_simulator import StochasticSEIRSimulator


def representative_fixed_interval(config: dict) -> int:
    """Choose the fixed-schedule baseline used in compact comparisons."""
    intervals = config["evaluation"].get("fixed_intervals", [7, 14, 21, 28])
    return 14 if 14 in intervals else int(intervals[0])


def representative_threshold(config: dict) -> float:
    """Choose the threshold baseline used in compact comparisons."""
    thresholds = config["evaluation"].get("thresholds", [0.001, 0.002, 0.003, 0.005])
    return 0.003 if 0.003 in thresholds else float(thresholds[0])


class Policy(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def action(self, state, info: dict) -> int:
        ...


def make_env(config: dict, budget: int | None = None, seed: int | None = None) -> RecalibrationEnv:
    """Create a recalibration environment from config."""
    simulator = StochasticSEIRSimulator(**config["simulator"])
    surrogate = DeterministicSEIRSurrogate(**config["surrogate"])
    env_cfg = config["environment"]
    env_budget = env_cfg["recalibration_budget"] if budget is None else budget
    env = RecalibrationEnv(
        true_simulator=simulator,
        surrogate_model=surrogate,
        termination_mode=env_cfg.get("termination_mode", "epidemic"),
        horizon=env_cfg["horizon"],
        max_steps=env_cfg.get("max_steps"),
        extinction_threshold=env_cfg.get("extinction_threshold", 1),
        extinction_patience=env_cfg.get("extinction_patience", 7),
        min_steps_before_extinction_check=env_cfg.get("min_steps_before_extinction_check", 14),
        forecast_horizon=env_cfg["forecast_horizon"],
        recalibration_budget=env_budget,
        error_metric=env_cfg["error_metric"],
        state_features=env_cfg.get("state_features"),
        error_history_length=env_cfg.get("error_history_length", 7),
        state_error_mode=env_cfg.get("state_error_mode", "immediate"),
        immediate_error_weight=env_cfg.get("immediate_error_weight", 0.3),
        path_error_weight=env_cfg.get("path_error_weight", 0.7),
        recalibration_config=config.get("recalibration"),
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate_policy(env: RecalibrationEnv, policy: Policy, episodes: int, seed: int) -> pd.DataFrame:
    """Run policy evaluation and return step-level trajectory rows."""
    rows = []
    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        policy.reset()
        info = {
            "t": 0,
            "remaining_budget": env.remaining_budget,
            "initial_budget": env.initial_budget,
            "days_since_recalibration": env.days_since_recalibration,
            "error": 0.0,
            "forecast_error": 0.0,
            "rolling_mean_forecast_error": 0.0,
            "rolling_error_trend": 0.0,
        }
        done = False
        while not done:
            if hasattr(policy, "action"):
                action = policy.action(observation, info)
            else:
                action = policy.act(observation)
            result = env.step(action)
            info = result.info
            rows.append(
                {
                    "policy": policy.name,
                    "episode": episode,
                    "t": int(info["t"]),
                    "decision_t": int(info["t"]) - 1,
                    "action": int(info["action"]),
                    "requested_recalibration": int(info.get("requested_recalibration", False)),
                    "recalibrated": int(info["recalibrated"]),
                    "recalibration_skipped": int(info["recalibration_skipped"]),
                    "remaining_budget": int(info["remaining_budget"]),
                    "initial_budget": int(info["initial_budget"]),
                    "budget_used": int(info["budget_used"]),
                    "budget_utilization": info["budget_utilization"],
                    "forecast_error": info["forecast_error"],
                    "matured_path_error": info.get("matured_path_error", float("nan")),
                    "matured_forecast_origin": info.get("matured_forecast_origin", -1),
                    "causal_mixed_error": info.get("causal_mixed_error", float("nan")),
                    "state_error_signal": info.get("state_error_signal", info["forecast_error"]),
                    "state_rolling_mean_error": info.get(
                        "state_rolling_mean_error", info["rolling_mean_forecast_error"]
                    ),
                    "pending_forecast_count": info.get("pending_forecast_count", 0),
                    "dropped_pending_transitions": info.get("dropped_pending_transitions", 0),
                    "cumulative_error": info["cumulative_error"],
                    "termination_reason": info["termination_reason"],
                    "active_infections": info["active_infections"],
                    "days_since_recalibration": info["days_since_recalibration"],
                    "true_beta": info["true_beta"],
                    "beta_hat": info["beta_hat"],
                    "recalibration_method": info["recalibration_method"],
                    "recalibration_old_beta": info["recalibration_old_beta"],
                    "recalibration_selected_beta": info["recalibration_selected_beta"],
                    "recalibration_applied_beta": info["recalibration_applied_beta"],
                    "recalibration_fit_loss": info["recalibration_fit_loss"],
                    "recalibration_valid_pairs": info["recalibration_valid_pairs"],
                    "true_I": info["true_I"],
                    "forecast_I": info["forecast_I"],
                    "true_S": info["true_state"]["S"],
                    "true_E": info["true_state"]["E"],
                    "true_R": info["true_state"]["R"],
                }
            )
            observation = result.observation
            done = result.done
    return pd.DataFrame(rows)


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

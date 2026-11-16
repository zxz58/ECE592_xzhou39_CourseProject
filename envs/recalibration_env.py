"""Oracle-only environment for budget-constrained surrogate recalibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.surrogate_model import DeterministicSEIRSurrogate
from envs.true_simulator import StochasticSEIRSimulator

WAIT = 0
RECALIBRATE = 1
DEFAULT_STATE_FEATURES = ["normalized_compartments", "recent_errors", "rolling_mean_forecast_error", "rolling_error_trend", "remaining_budget", "days_since_recalibration"]
VALID_STATE_FEATURES = set(DEFAULT_STATE_FEATURES) | {"last_forecast_error"}


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done
        yield self.info


class RecalibrationEnv:
    """Binary-action environment with exact-current-beta oracle updates."""

    action_size = 2

    def __init__(
        self,
        true_simulator: StochasticSEIRSimulator,
        surrogate_model: DeterministicSEIRSurrogate,
        termination_mode: str,
        horizon: int,
        max_steps: int,
        extinction_threshold: float = 1.0,
        extinction_patience: int = 7,
        min_steps_before_extinction_check: int = 14,
        forecast_horizon: int = 7,
        recalibration_budget: int = 6,
        error_metric: str = "mae",
        state_features: list[str] | None = None,
        error_history_length: int = 14,
        immediate_error_weight: float = 0.3,
        path_error_weight: float = 0.7,
    ) -> None:
        self.true_simulator = true_simulator
        self.surrogate_model = surrogate_model
        self.termination_mode = str(termination_mode)
        self.horizon = int(horizon)
        self.max_steps = int(max_steps)
        self.extinction_threshold = float(extinction_threshold)
        self.extinction_patience = int(extinction_patience)
        self.min_steps_before_extinction_check = int(min_steps_before_extinction_check)
        self.forecast_horizon = int(forecast_horizon)
        self.initial_budget = int(recalibration_budget)
        self.budget = self.initial_budget
        self.error_metric = str(error_metric)
        self.state_features = list(state_features or DEFAULT_STATE_FEATURES)
        self.error_history_length = int(error_history_length)
        self.immediate_error_weight = float(immediate_error_weight)
        self.path_error_weight = float(path_error_weight)
        self._validate()
        self.remaining_budget = self.initial_budget
        self.days_since_recalibration = 0
        self.extinction_counter = 0
        self.error_history: list[float] = []
        self.state_error_history: list[float] = []
        self.history: list[dict[str, float]] = []
        self.pending_forecasts: list[dict[str, Any]] = []
        self.cumulative_error = 0.0

    def _validate(self) -> None:
        if self.termination_mode not in {"epidemic", "fixed_horizon"}:
            raise ValueError("termination_mode must be epidemic or fixed_horizon")
        if min(self.horizon, self.max_steps, self.forecast_horizon) <= 0:
            raise ValueError("horizon, max_steps, and forecast_horizon must be positive")
        if self.initial_budget < 0 or self.error_history_length <= 0:
            raise ValueError("budget must be nonnegative and history length positive")
        if self.error_metric not in {"mae", "mse", "infectious_mse"}:
            raise ValueError("unsupported error metric")
        if not np.isclose(self.immediate_error_weight + self.path_error_weight, 1.0):
            raise ValueError("error weights must sum to one")
        unknown = set(self.state_features) - VALID_STATE_FEATURES
        if unknown:
            raise ValueError(f"unknown state features: {sorted(unknown)}")
        required = {"normalized_compartments", "remaining_budget", "days_since_recalibration"}
        if not required.issubset(self.state_features):
            raise ValueError(f"state features must include {sorted(required)}")

    @property
    def observation_size(self) -> int:
        return sum(4 if f == "normalized_compartments" else self.error_history_length if f == "recent_errors" else 1 for f in self.state_features)

    def reset(self, seed: int | None = None) -> np.ndarray:
        state = self.true_simulator.reset(seed=seed)
        self.surrogate_model.reset()
        self.remaining_budget = self.initial_budget
        self.days_since_recalibration = 0
        self.extinction_counter = 0
        self.error_history = [0.0] * self.error_history_length
        self.state_error_history = [0.0] * self.error_history_length
        self.history = [state.copy()]
        self.pending_forecasts = []
        self.cumulative_error = 0.0
        return self._observation(state)

    def step(self, action: int) -> StepResult:
        action = int(action)
        if action not in (WAIT, RECALIBRATE):
            raise ValueError("action must be 0 or 1")
        origin = self.true_simulator.get_state()
        requested = action == RECALIBRATE
        recalibrated = requested and self.remaining_budget > 0
        old_beta = float(self.surrogate_model.beta_hat)
        true_beta = float(self.true_simulator.get_true_beta())
        if recalibrated:
            self.surrogate_model.recalibrate(true_beta=true_beta)
            self.remaining_budget -= 1
            self.days_since_recalibration = 0
        else:
            self.days_since_recalibration += 1

        forecast_path = self.surrogate_model.forecast(origin, self.forecast_horizon)
        self.pending_forecasts.append({"origin_t": int(origin["t"]), "path": [value.copy() for value in forecast_path]})
        truth = self.true_simulator.step()
        immediate_error = self._compute_error(forecast_path[0], truth)
        self.pending_forecasts[-1]["immediate_error"] = immediate_error
        self.cumulative_error += immediate_error
        self._push(self.error_history, immediate_error)
        self.history.append(truth.copy())
        matured = self._mature(int(truth["t"]))
        state_error = immediate_error if matured is None else float(matured["mixed_error"])
        self._push(self.state_error_history, state_error)
        done, reason = self._check_done(truth)
        immediate_stats = self._stats(self.error_history)
        state_stats = self._stats(self.state_error_history)
        info = {
            "decision_t": int(origin["t"]), "t": int(truth["t"]), "action": action,
            "requested_recalibration": requested, "recalibrated": recalibrated,
            "recalibration_skipped": requested and not recalibrated,
            "remaining_budget": self.remaining_budget, "initial_budget": self.initial_budget,
            "budget_used": self.initial_budget - self.remaining_budget,
            "budget_utilization": (self.initial_budget - self.remaining_budget) / max(self.initial_budget, 1),
            "days_since_recalibration": self.days_since_recalibration,
            "true_beta": true_beta, "beta_hat": float(self.surrogate_model.beta_hat),
            "recalibration_method": "oracle", "recalibration_old_beta": old_beta,
            "recalibration_selected_beta": true_beta if recalibrated else np.nan,
            "recalibration_applied_beta": float(self.surrogate_model.beta_hat),
            "forecast_error": immediate_error, "cumulative_error": self.cumulative_error,
            "rolling_mean_forecast_error": immediate_stats[1], "rolling_error_trend": immediate_stats[2],
            "state_error_signal": state_error, "state_rolling_mean_error": state_stats[1],
            "state_rolling_error_trend": state_stats[2],
            "matured_forecast_origin": -1 if matured is None else int(matured["origin_t"]),
            "matured_path_error": np.nan if matured is None else float(matured["path_error"]),
            "causal_mixed_error": np.nan if matured is None else float(matured["mixed_error"]),
            "surrogate_forecast": forecast_path, "true_state": truth.copy(),
            "active_infections": float(truth["E"] + truth["I"]),
            "pending_forecast_count": len(self.pending_forecasts),
            "dropped_pending_transitions": len(self.pending_forecasts) if done else 0,
            "termination_reason": reason, "done": done,
        }
        return StepResult(self._observation(truth), -state_error, done, info)

    def _push(self, history: list[float], value: float) -> None:
        history.append(float(value))
        del history[0]

    def _compute_error(self, forecast: dict[str, float], truth: dict[str, float]) -> float:
        population = float(self.true_simulator.population)
        if self.error_metric == "infectious_mse":
            return float(((forecast["I"] - truth["I"]) / population) ** 2)
        errors = np.asarray([(forecast[name] - truth[name]) / population for name in "SEIR"])
        return float(np.mean(errors**2)) if self.error_metric == "mse" else float(np.mean(np.abs(errors)))

    def _path_mae(self, forecast: dict[str, float], truth: dict[str, float]) -> float:
        population = float(self.true_simulator.population)
        return float(np.mean([abs(forecast[name] - truth[name]) / population for name in "SEIR"]))

    def _mature(self, current_t: int) -> dict[str, float] | None:
        if not self.pending_forecasts or int(self.pending_forecasts[0]["origin_t"]) + self.forecast_horizon != current_t:
            return None
        pending = self.pending_forecasts.pop(0)
        errors = [self._path_mae(forecast, self.history[int(pending["origin_t"]) + offset]) for offset, forecast in enumerate(pending["path"], start=1)]
        path_error = float(np.mean(errors))
        return {
            "origin_t": int(pending["origin_t"]), "path_error": path_error,
            "mixed_error": self.immediate_error_weight * float(pending["immediate_error"]) + self.path_error_weight * path_error,
        }

    def _stats(self, history: list[float]) -> tuple[float, float, float]:
        midpoint = max(len(history) // 2, 1)
        return float(history[-1]), float(np.mean(history)), float(np.mean(history[midpoint:]) - np.mean(history[:midpoint]))

    def _observation(self, state: dict[str, float]) -> np.ndarray:
        population = float(self.true_simulator.population)
        last, mean, trend = self._stats(self.state_error_history)
        values: list[float] = []
        for feature in self.state_features:
            if feature == "normalized_compartments": values.extend(float(state[name]) / population for name in "SEIR")
            elif feature == "recent_errors": values.extend(self.state_error_history)
            elif feature == "last_forecast_error": values.append(last)
            elif feature == "rolling_mean_forecast_error": values.append(mean)
            elif feature == "rolling_error_trend": values.append(trend)
            elif feature == "remaining_budget": values.append(self.remaining_budget / max(self.initial_budget, 1))
            elif feature == "days_since_recalibration": values.append(min(self.days_since_recalibration, self.max_steps) / self.max_steps)
        return np.asarray(values, dtype=np.float32)

    def _check_done(self, state: dict[str, float]) -> tuple[bool, str | None]:
        t = int(state["t"])
        if self.termination_mode == "fixed_horizon":
            return (True, "fixed_horizon") if t >= self.horizon else (False, None)
        active = float(state["E"] + state["I"])
        if t >= self.min_steps_before_extinction_check and active <= self.extinction_threshold:
            self.extinction_counter += 1
        else:
            self.extinction_counter = 0
        if self.extinction_counter >= self.extinction_patience:
            return True, "extinction"
        return (True, "max_steps") if t >= self.max_steps else (False, None)

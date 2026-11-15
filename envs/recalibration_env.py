"""Gym-like environment for budget-constrained surrogate recalibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from envs.surrogate_model import DeterministicSEIRSurrogate
from envs.true_simulator import StochasticSEIRSimulator

WAIT = 0
RECALIBRATE = 1
DEFAULT_STATE_FEATURES = [
    "normalized_compartments",
    "recent_errors",
    "rolling_mean_forecast_error",
    "rolling_error_trend",
    "remaining_budget",
    "days_since_recalibration",
]
SCALAR_STATE_FEATURES = {
    "last_forecast_error",
    "rolling_mean_forecast_error",
    "rolling_error_trend",
    "remaining_budget",
    "days_since_recalibration",
}
VALID_STATE_FEATURES = {"normalized_compartments", "recent_errors"} | SCALAR_STATE_FEATURES


@dataclass
class StepResult:
    """Small container for Gym-like step outputs."""

    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]

    def __iter__(self):
        """Allow unpacking as next_state, reward, done, info."""
        yield self.observation
        yield self.reward
        yield self.done
        yield self.info


@dataclass
class RecalibrationResult:
    """Diagnostics from one recalibration attempt."""

    method: str
    old_beta: float
    selected_beta: float
    applied_beta: float
    fit_loss: float
    valid_pairs: int
    skipped: bool


class RecalibrationEnv:
    """Binary-action environment: wait or recalibrate the surrogate model."""

    action_size = 2

    def __init__(
        self,
        true_simulator: StochasticSEIRSimulator | None = None,
        surrogate_model: DeterministicSEIRSurrogate | None = None,
        termination_mode: str = "epidemic",
        horizon: int | None = None,
        max_steps: int | None = None,
        extinction_threshold: float = 1.0,
        extinction_patience: int = 7,
        min_steps_before_extinction_check: int = 14,
        forecast_horizon: int = 1,
        recalibration_budget: int | None = None,
        error_metric: str = "mae",
        state_features: list[str] | None = None,
        error_history_length: int = 14,
        state_error_mode: str = "immediate",
        immediate_error_weight: float = 0.3,
        path_error_weight: float = 0.7,
        recalibration_config: dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        # Backward-compatible aliases used by the first prototype scripts.
        if true_simulator is None:
            true_simulator = legacy_kwargs.pop("simulator", None)
        if surrogate_model is None:
            surrogate_model = legacy_kwargs.pop("surrogate", None)
        if recalibration_budget is None:
            recalibration_budget = legacy_kwargs.pop("budget", None)
        if legacy_kwargs:
            raise TypeError(f"Unexpected RecalibrationEnv arguments: {sorted(legacy_kwargs)}")
        if true_simulator is None or surrogate_model is None:
            raise ValueError("true_simulator and surrogate_model are required")

        self.true_simulator = true_simulator
        self.surrogate_model = surrogate_model
        self.simulator = true_simulator
        self.surrogate = surrogate_model
        self.termination_mode = termination_mode
        self.horizon = int(horizon or true_simulator.episode_length)
        self.max_steps = int(max_steps or self.horizon)
        self.extinction_threshold = float(extinction_threshold)
        self.extinction_patience = int(extinction_patience)
        self.min_steps_before_extinction_check = int(min_steps_before_extinction_check)
        self.forecast_horizon = int(forecast_horizon)
        self.initial_budget = int(recalibration_budget if recalibration_budget is not None else 0)
        self.budget = self.initial_budget
        self.error_metric = error_metric
        self.error_history_length = int(error_history_length)
        self.state_error_mode = str(state_error_mode)
        self.immediate_error_weight = float(immediate_error_weight)
        self.path_error_weight = float(path_error_weight)
        self.recalibration_config = dict(recalibration_config or {})
        self.recalibration_method = str(self.recalibration_config.get("method", "oracle"))
        self.recalibration_noise_type = str(self.recalibration_config.get("noise_type", "gaussian"))
        self.recalibration_noise_std = float(self.recalibration_config.get("noise_std", 0.0))
        clip_bounds = self.recalibration_config.get("clip_bounds", [0.01, 2.0])
        self.recalibration_clip_bounds = (float(clip_bounds[0]), float(clip_bounds[1]))
        beta_bounds = self.recalibration_config.get("beta_bounds", self.recalibration_clip_bounds)
        self.recalibration_beta_bounds = (float(beta_bounds[0]), float(beta_bounds[1]))
        self.recalibration_fit_window = int(
            self.recalibration_config.get("fit_window", self.surrogate_model.fit_window)
        )
        self.recalibration_grid_size = int(self.recalibration_config.get("grid_size", 64))
        self.recalibration_min_pairs = int(self.recalibration_config.get("min_pairs", 1))
        self.recalibration_update_gain = float(self.recalibration_config.get("update_gain", 0.5))
        self.recalibration_moving_average_window = int(self.recalibration_config.get("moving_average_window", 3))
        self.partial_update_base_method = str(self.recalibration_config.get("base_method", "moment_match"))
        # Default state is D_raw_mean_trend with configurable history length K.
        self.state_features = list(state_features or DEFAULT_STATE_FEATURES)
        self._validate_configuration()

        self.remaining_budget = self.initial_budget
        self.days_since_recalibration = 0
        self.extinction_counter = 0
        self.error_history = [0.0] * self.error_history_length
        self.state_error_history = [0.0] * self.error_history_length
        self.cumulative_error = 0.0
        self.history: list[dict[str, float]] = []
        self.pending_forecasts: list[dict[str, Any]] = []
        self.recalibration_rng = np.random.default_rng()
        self.beta_estimate_history: list[float] = []
        self.last_recalibration_result = self._empty_recalibration_result()

    @property
    def observation_size(self) -> int:
        """Return the size of the configured observation vector."""
        size = 0
        for feature in self.state_features:
            if feature == "normalized_compartments":
                size += 4
            elif feature == "recent_errors":
                size += self.error_history_length
            else:
                size += 1
        return size

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset simulator, surrogate, budget state, and diagnostics."""
        state = self.true_simulator.reset(seed=seed)
        self.surrogate_model.reset()
        self.remaining_budget = self.initial_budget
        self.days_since_recalibration = 0
        self.extinction_counter = 0
        self.error_history = [0.0] * self.error_history_length
        self.state_error_history = [0.0] * self.error_history_length
        self.cumulative_error = 0.0
        self.history = [state.copy()]
        self.pending_forecasts = []
        self.recalibration_rng = np.random.default_rng(None if seed is None else seed + 1_000_003)
        self.beta_estimate_history = []
        self.last_recalibration_result = self._empty_recalibration_result()
        return self._observation(state)

    def step(self, action: int) -> StepResult:
        """Apply action, forecast from current truth, advance truth, score error."""
        current_true_state = self.true_simulator.get_state()
        action = int(action)
        if action not in (WAIT, RECALIBRATE):
            raise ValueError("action must be 0 (WAIT) or 1 (RECALIBRATE)")

        requested_recalibration = action == RECALIBRATE
        recalibrated = False
        if requested_recalibration and self.remaining_budget > 0:
            # Uses only current beta before truth advances, avoiding future leakage.
            self.last_recalibration_result = self._apply_recalibration(current_true_state)
            recalibrated = not self.last_recalibration_result.skipped
            if recalibrated:
                self.remaining_budget -= 1
                self.days_since_recalibration = 0
        else:
            self.last_recalibration_result = self._empty_recalibration_result()

        if not recalibrated:
            self.days_since_recalibration += 1

        forecast_path = self.surrogate_model.forecast(current_true_state, self.forecast_horizon)
        one_step_forecast = forecast_path[0]
        self.pending_forecasts.append(
            {
                "origin_t": int(current_true_state["t"]),
                "forecast_path": [state.copy() for state in forecast_path],
            }
        )
        next_true_state = self.true_simulator.step()

        error = self._compute_error(one_step_forecast, next_true_state)
        reward = -error
        self.cumulative_error += error
        self._push_error(error)
        self.history.append(next_true_state.copy())
        self.pending_forecasts[-1]["immediate_error"] = float(error)
        matured = self._mature_forecast_path(int(next_true_state["t"]))
        matured_path_error = float(matured["path_error"]) if matured is not None else np.nan
        if self.state_error_mode == "causal_mixed" and matured is not None:
            state_error = self.immediate_error_weight * error + self.path_error_weight * matured_path_error
        else:
            state_error = error
        self._push_state_error(state_error)

        done, termination_reason = self._check_done(next_true_state)
        observation = self._observation(next_true_state)
        info = self._build_info(
            next_true_state=next_true_state,
            forecast_path=forecast_path,
            one_step_forecast=one_step_forecast,
            error=error,
            reward=reward,
            action=action,
            recalibrated=recalibrated,
            matured=matured,
            state_error=state_error,
            done=done,
            termination_reason=termination_reason,
        )

        return StepResult(observation, reward, done, info)

    def _validate_configuration(self) -> None:
        """Validate options that affect state and reward definitions."""
        if self.termination_mode not in {"epidemic", "fixed_horizon"}:
            raise ValueError("termination_mode must be 'epidemic' or 'fixed_horizon'")
        if self.horizon <= 0 or self.max_steps <= 0:
            raise ValueError("horizon and max_steps must be positive")
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if self.initial_budget < 0:
            raise ValueError("recalibration_budget must be nonnegative")
        methods = {"oracle", "oracle_noisy", "moment_match", "grid_fit", "moving_average", "partial_update"}
        if self.recalibration_method not in methods:
            raise ValueError(f"recalibration.method must be one of {sorted(methods)}")
        if self.recalibration_noise_type != "gaussian":
            raise ValueError("recalibration.noise_type must be 'gaussian'")
        if self.recalibration_noise_std < 0.0:
            raise ValueError("recalibration.noise_std must be nonnegative")
        if self.recalibration_grid_size <= 1:
            raise ValueError("recalibration.grid_size must be greater than 1")
        if self.recalibration_fit_window <= 0:
            raise ValueError("recalibration.fit_window must be positive")
        if self.recalibration_min_pairs <= 0:
            raise ValueError("recalibration.min_pairs must be positive")
        if self.recalibration_moving_average_window <= 0:
            raise ValueError("recalibration.moving_average_window must be positive")
        if not 0.0 <= self.recalibration_update_gain <= 1.0:
            raise ValueError("recalibration.update_gain must be in [0, 1]")
        for bounds_name, bounds in {
            "clip_bounds": self.recalibration_clip_bounds,
            "beta_bounds": self.recalibration_beta_bounds,
        }.items():
            lower, upper = bounds
            if lower < 0.0 or upper <= lower:
                raise ValueError(f"recalibration.{bounds_name} must be [lower, upper] with 0 <= lower < upper")
        if self.partial_update_base_method not in {"oracle", "oracle_noisy", "moment_match", "grid_fit", "moving_average"}:
            raise ValueError("recalibration.base_method is not supported for partial_update")
        if self.error_history_length <= 0:
            raise ValueError("error_history_length must be positive")
        if self.state_error_mode not in {"immediate", "causal_mixed"}:
            raise ValueError("state_error_mode must be 'immediate' or 'causal_mixed'")
        if self.immediate_error_weight < 0.0 or self.path_error_weight < 0.0:
            raise ValueError("error weights must be nonnegative")
        if not np.isclose(self.immediate_error_weight + self.path_error_weight, 1.0):
            raise ValueError("immediate_error_weight + path_error_weight must equal 1")
        if self.extinction_patience <= 0:
            raise ValueError("extinction_patience must be positive")
        valid_metrics = {"mse", "mae", "infectious_mse"}
        if self.error_metric not in valid_metrics:
            raise ValueError(f"error_metric must be one of {sorted(valid_metrics)}")
        unknown_features = sorted(set(self.state_features) - VALID_STATE_FEATURES)
        if unknown_features:
            raise ValueError(f"Unknown state feature(s): {unknown_features}")
        if "normalized_compartments" not in self.state_features:
            raise ValueError("state_features must include normalized_compartments")
        if "remaining_budget" not in self.state_features:
            raise ValueError("state_features must include remaining_budget")
        if "days_since_recalibration" not in self.state_features:
            raise ValueError("state_features must include days_since_recalibration")

    def _empty_recalibration_result(self) -> RecalibrationResult:
        """Return a placeholder result for steps without recalibration attempts."""
        beta = float(getattr(self.surrogate_model, "beta_hat", np.nan))
        return RecalibrationResult(self.recalibration_method, beta, np.nan, beta, np.nan, 0, False)

    def _apply_recalibration(self, current_true_state: dict[str, float]) -> RecalibrationResult:
        """Select and apply beta according to the configured recalibration method."""
        old_beta = float(self.surrogate_model.beta_hat)
        result = self._select_recalibration_beta(self.recalibration_method, current_true_state, old_beta)
        if not result.skipped:
            self.surrogate_model.recalibrate(true_beta=result.applied_beta)
        return result

    def _select_recalibration_beta(
        self,
        method: str,
        current_true_state: dict[str, float],
        old_beta: float,
    ) -> RecalibrationResult:
        """Return the beta selected by a method without applying future-state information."""
        if method == "oracle":
            selected = float(self.true_simulator.get_true_beta())
            applied = self._clip_beta(selected, self.recalibration_clip_bounds)
            return RecalibrationResult(method, old_beta, selected, applied, np.nan, 1, False)

        if method == "oracle_noisy":
            true_beta = float(self.true_simulator.get_true_beta())
            noise = 0.0
            if self.recalibration_noise_std > 0.0:
                noise = float(self.recalibration_rng.normal(0.0, self.recalibration_noise_std))
            selected = true_beta + noise
            applied = self._clip_beta(selected, self.recalibration_clip_bounds)
            return RecalibrationResult(method, old_beta, selected, applied, np.nan, 1, False)

        if method == "moment_match":
            estimates = self._moment_beta_estimates(self.history, self.recalibration_fit_window)
            if len(estimates) < self.recalibration_min_pairs:
                return RecalibrationResult(method, old_beta, old_beta, old_beta, np.nan, len(estimates), True)
            selected = float(np.mean(estimates))
            applied = self._clip_beta(selected, self.recalibration_beta_bounds)
            return RecalibrationResult(method, old_beta, selected, applied, np.nan, len(estimates), False)

        if method == "moving_average":
            estimates = self._moment_beta_estimates(self.history, self.recalibration_fit_window)
            if len(estimates) < self.recalibration_min_pairs:
                return RecalibrationResult(method, old_beta, old_beta, old_beta, np.nan, len(estimates), True)
            current_estimate = float(np.mean(estimates))
            self.beta_estimate_history.append(current_estimate)
            selected = float(np.mean(self.beta_estimate_history[-self.recalibration_moving_average_window :]))
            applied = self._clip_beta(selected, self.recalibration_beta_bounds)
            return RecalibrationResult(method, old_beta, selected, applied, np.nan, len(estimates), False)

        if method == "grid_fit":
            return self._grid_fit_result(old_beta)

        if method == "partial_update":
            base = self._select_recalibration_beta(self.partial_update_base_method, current_true_state, old_beta)
            if base.skipped:
                return RecalibrationResult(method, old_beta, old_beta, old_beta, base.fit_loss, base.valid_pairs, True)
            applied = self._clip_beta(
                old_beta + self.recalibration_update_gain * (base.selected_beta - old_beta),
                self.recalibration_beta_bounds,
            )
            return RecalibrationResult(method, old_beta, base.selected_beta, applied, base.fit_loss, base.valid_pairs, False)

        raise ValueError(f"Unsupported recalibration method: {method}")

    def _clip_beta(self, beta: float, bounds: tuple[float, float]) -> float:
        """Clip beta to configured bounds."""
        return float(np.clip(beta, bounds[0], bounds[1]))

    def _moment_beta_estimates(self, history: list[dict[str, float]], fit_window: int) -> list[float]:
        """Estimate beta from causal state transitions in history."""
        states = history[-fit_window - 1 :]
        estimates: list[float] = []
        for current, nxt in zip(states[:-1], states[1:]):
            population = current["S"] + current["E"] + current["I"] + current["R"]
            if population <= 0 or current["S"] <= 0 or current["I"] <= 0:
                continue
            new_infectious = self.surrogate_model.sigma_hat * current["E"]
            new_exposed = nxt["E"] - current["E"] + new_infectious
            exposure_rate = np.clip(new_exposed / current["S"], 1e-9, 0.999999)
            beta = -np.log(1.0 - exposure_rate) * population / current["I"]
            if np.isfinite(beta):
                estimates.append(self._clip_beta(float(beta), self.recalibration_beta_bounds))
        return estimates

    def _grid_fit_result(self, old_beta: float) -> RecalibrationResult:
        """Fit beta by grid search over historical one-step forecast errors."""
        states = self.history[-self.recalibration_fit_window - 1 :]
        pairs = [
            (current, nxt)
            for current, nxt in zip(states[:-1], states[1:])
            if current["S"] + current["E"] + current["I"] + current["R"] > 0 and current["I"] >= 0
        ]
        if len(pairs) < self.recalibration_min_pairs:
            return RecalibrationResult("grid_fit", old_beta, old_beta, old_beta, np.nan, len(pairs), True)

        candidates = np.linspace(self.recalibration_beta_bounds[0], self.recalibration_beta_bounds[1], self.recalibration_grid_size)
        losses = []
        population = float(self.true_simulator.population)
        for candidate in candidates:
            pair_losses = []
            for current, nxt in pairs:
                forecast = self._one_step_forecast_with_beta(current, float(candidate))
                errors = np.array(
                    [
                        (forecast["S"] - nxt["S"]) / population,
                        (forecast["E"] - nxt["E"]) / population,
                        (forecast["I"] - nxt["I"]) / population,
                        (forecast["R"] - nxt["R"]) / population,
                    ],
                    dtype=float,
                )
                pair_losses.append(float(np.mean(np.abs(errors))))
            losses.append(float(np.mean(pair_losses)))
        best_index = int(np.argmin(losses))
        selected = float(candidates[best_index])
        applied = self._clip_beta(selected, self.recalibration_beta_bounds)
        return RecalibrationResult("grid_fit", old_beta, selected, applied, float(losses[best_index]), len(pairs), False)

    def _one_step_forecast_with_beta(self, state: dict[str, float], beta: float) -> dict[str, float]:
        """Forecast one step from a historical state using a candidate beta."""
        population = state["S"] + state["E"] + state["I"] + state["R"]
        infection_prob = float(np.clip(1.0 - np.exp(-beta * state["I"] / population), 0.0, 1.0))
        new_exposed = state["S"] * infection_prob
        new_infectious = state["E"] * self.surrogate_model.sigma_hat
        new_recovered = state["I"] * self.surrogate_model.gamma_hat
        return {
            "S": state["S"] - new_exposed,
            "E": state["E"] + new_exposed - new_infectious,
            "I": state["I"] + new_infectious - new_recovered,
            "R": state["R"] + new_recovered,
        }

    def _check_done(self, state: dict[str, float]) -> tuple[bool, str | None]:
        """Check fixed-horizon or epidemic-extinction termination."""
        t = int(state["t"])
        active = float(state["E"] + state["I"])
        if self.termination_mode == "fixed_horizon":
            if t >= self.horizon:
                return True, "fixed_horizon"
            return False, None

        if active <= self.extinction_threshold and t >= self.min_steps_before_extinction_check:
            self.extinction_counter += 1
        else:
            self.extinction_counter = 0
        if self.extinction_counter >= self.extinction_patience:
            return True, "extinction"
        if t >= self.max_steps:
            return True, "max_steps"
        return False, None

    def _compute_error(self, forecast: dict[str, float], truth: dict[str, float]) -> float:
        """Compute normalized one-step forecast error."""
        population = self.true_simulator.population
        if self.error_metric == "infectious_mse":
            return float(((forecast["I"] - truth["I"]) / population) ** 2)

        errors = np.array(
            [
                (forecast["S"] - truth["S"]) / population,
                (forecast["E"] - truth["E"]) / population,
                (forecast["I"] - truth["I"]) / population,
                (forecast["R"] - truth["R"]) / population,
            ],
            dtype=float,
        )
        if self.error_metric == "mse":
            return float(np.mean(errors**2))
        return float(np.mean(np.abs(errors)))

    def _compute_normalized_mae(self, forecast: dict[str, float], truth: dict[str, float]) -> float:
        """Compute normalized SEIR MAE independently of the immediate-error metric."""
        population = float(self.true_simulator.population)
        return float(
            np.mean(
                [
                    abs(float(forecast[name]) - float(truth[name])) / population
                    for name in ("S", "E", "I", "R")
                ]
            )
        )

    def _mature_forecast_path(self, current_t: int) -> dict[str, float] | None:
        """Return the oldest forecast only when all of its truths are observed."""
        if not self.pending_forecasts:
            return None
        pending = self.pending_forecasts[0]
        origin_t = int(pending["origin_t"])
        if origin_t + self.forecast_horizon != current_t:
            return None
        pending = self.pending_forecasts.pop(0)
        path_errors = [
            self._compute_normalized_mae(forecast, self.history[origin_t + offset])
            for offset, forecast in enumerate(pending["forecast_path"], start=1)
        ]
        path_error = float(np.mean(path_errors))
        mixed_error = (
            self.immediate_error_weight * float(pending["immediate_error"])
            + self.path_error_weight * path_error
        )
        return {
            "origin_t": origin_t,
            "path_error": path_error,
            "mixed_error": mixed_error,
        }

    def _push_error(self, error: float) -> None:
        """Maintain a fixed-length recent error history."""
        self.error_history.append(float(error))
        self.error_history = self.error_history[-self.error_history_length :]

    def _push_state_error(self, error: float) -> None:
        """Maintain the causal error signal exposed to the policy state."""
        self.state_error_history.append(float(error))
        self.state_error_history = self.state_error_history[-self.error_history_length :]

    def _error_stats(self, history: list[float] | None = None) -> tuple[float, float, float]:
        """Return last error, rolling mean, and simple trend."""
        history = self.error_history if history is None else history
        last_error = float(history[-1])
        rolling_mean = float(np.mean(history))
        if len(history) < 2:
            return last_error, rolling_mean, 0.0
        midpoint = max(len(history) // 2, 1)
        earlier = float(np.mean(history[:midpoint]))
        later = float(np.mean(history[midpoint:]))
        return last_error, rolling_mean, later - earlier

    def _observation(self, state: dict[str, float]) -> np.ndarray:
        """Build the configured state feature vector."""
        population = self.true_simulator.population
        last_error, rolling_mean, rolling_trend = self._error_stats(self.state_error_history)
        values = []
        for feature in self.state_features:
            if feature == "normalized_compartments":
                values.extend(
                    [
                        state["S"] / population,
                        state["E"] / population,
                        state["I"] / population,
                        state["R"] / population,
                    ]
                )
            elif feature == "recent_errors":
                values.extend(self.state_error_history)
            elif feature == "last_forecast_error":
                values.append(last_error)
            elif feature == "rolling_mean_forecast_error":
                values.append(rolling_mean)
            elif feature == "rolling_error_trend":
                values.append(rolling_trend)
            elif feature == "remaining_budget":
                values.append(self.remaining_budget / max(self.initial_budget, 1))
            elif feature == "days_since_recalibration":
                values.append(min(self.days_since_recalibration, self.max_steps) / self.max_steps)
        return np.array(values, dtype=np.float32)

    def _build_info(
        self,
        next_true_state: dict[str, float],
        forecast_path: list[dict[str, float]],
        one_step_forecast: dict[str, float],
        error: float,
        reward: float,
        action: int,
        recalibrated: bool,
        matured: dict[str, float] | None,
        state_error: float,
        done: bool,
        termination_reason: str | None,
    ) -> dict[str, Any]:
        """Collect diagnostics for logging and policies."""
        last_error, rolling_mean, rolling_trend = self._error_stats()
        state_last_error, state_rolling_mean, state_rolling_trend = self._error_stats(
            self.state_error_history
        )
        active = float(next_true_state["E"] + next_true_state["I"])
        matured_path_error = float(matured["path_error"]) if matured is not None else np.nan
        matured_origin = int(matured["origin_t"]) if matured is not None else -1
        causal_mixed_error = float(matured["mixed_error"]) if matured is not None else np.nan
        return {
            "t": int(next_true_state["t"]),
            "true_state": next_true_state.copy(),
            "surrogate_forecast": [state.copy() for state in forecast_path],
            "error": float(error),
            "forecast_error": float(error),
            "last_forecast_error": last_error,
            "rolling_mean_forecast_error": rolling_mean,
            "rolling_error_trend": rolling_trend,
            "state_error_signal": float(state_error),
            "state_last_error": state_last_error,
            "state_rolling_mean_error": state_rolling_mean,
            "state_rolling_error_trend": state_rolling_trend,
            "matured_path_error": matured_path_error,
            "matured_forecast_origin": matured_origin,
            "causal_mixed_error": causal_mixed_error,
            "pending_forecast_count": len(self.pending_forecasts),
            "dropped_pending_transitions": len(self.pending_forecasts) if done else 0,
            "reward": float(reward),
            "action": action,
            "requested_recalibration": bool(action == RECALIBRATE),
            "recalibrated": bool(recalibrated),
            "remaining_budget": int(self.remaining_budget),
            "initial_budget": int(self.initial_budget),
            "budget_used": int(self.initial_budget - self.remaining_budget),
            "budget_utilization": (self.initial_budget - self.remaining_budget) / max(self.initial_budget, 1),
            "days_since_recalibration": int(self.days_since_recalibration),
            "cumulative_error": float(self.cumulative_error),
            "true_beta": float(self.true_simulator.get_true_beta()),
            "beta_hat": float(self.surrogate_model.beta_hat),
            "recalibration_result": asdict(self.last_recalibration_result),
            "recalibration_method": self.last_recalibration_result.method,
            "recalibration_old_beta": float(self.last_recalibration_result.old_beta),
            "recalibration_selected_beta": float(self.last_recalibration_result.selected_beta),
            "recalibration_applied_beta": float(self.last_recalibration_result.applied_beta),
            "recalibration_fit_loss": float(self.last_recalibration_result.fit_loss),
            "recalibration_valid_pairs": int(self.last_recalibration_result.valid_pairs),
            "recalibration_skipped": bool(self.last_recalibration_result.skipped),
            "true_I": float(next_true_state["I"]),
            "forecast_I": float(one_step_forecast["I"]),
            "active_infections": active,
            "done": bool(done),
            "termination_reason": termination_reason,
            "extinction_counter": int(self.extinction_counter),
        }

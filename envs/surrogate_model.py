"""Deterministic SEIR surrogate model used for forecasting."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np


class DeterministicSEIRSurrogate:
    """Simple deterministic SEIR forecaster with recalibratable beta."""

    def __init__(
        self,
        beta_hat: float = 0.30,
        sigma_hat: float = 1 / 5.2,
        gamma_hat: float = 1 / 7,
        fit_window: int = 14,
    ) -> None:
        self.initial_beta_hat = float(beta_hat)
        self.beta_hat = float(beta_hat)
        self.sigma_hat = float(sigma_hat)
        self.gamma_hat = float(gamma_hat)
        self.fit_window = int(fit_window)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate deterministic SEIR parameters."""
        if self.beta_hat < 0.0:
            raise ValueError("beta_hat must be nonnegative")
        for name, value in {"sigma_hat": self.sigma_hat, "gamma_hat": self.gamma_hat}.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.fit_window <= 0:
            raise ValueError("fit_window must be positive")

    def reset(self) -> None:
        """Restore initial surrogate parameters."""
        self.beta_hat = self.initial_beta_hat

    def fit(self, history: Iterable[dict[str, float]]) -> float:
        """Estimate beta from recent state transitions by moment matching."""
        states = list(history)[-self.fit_window - 1 :]
        estimates: list[float] = []
        for current, nxt in zip(states[:-1], states[1:]):
            population = current["S"] + current["E"] + current["I"] + current["R"]
            if current["S"] <= 0 or current["I"] <= 0:
                continue
            new_infectious = self.sigma_hat * current["E"]
            new_exposed = nxt["E"] - current["E"] + new_infectious
            exposure_rate = np.clip(new_exposed / current["S"], 1e-9, 0.999999)
            beta = -np.log(1.0 - exposure_rate) * population / current["I"]
            if np.isfinite(beta):
                estimates.append(float(beta))
        if estimates:
            self.beta_hat = float(np.clip(np.mean(estimates), 0.01, 2.0))
        return self.beta_hat

    def recalibrate(
        self,
        true_beta: float | None = None,
        history: Iterable[dict[str, float]] | None = None,
    ) -> float:
        """Update beta with oracle truth when available, otherwise fit history."""
        if true_beta is not None:
            self.beta_hat = float(true_beta)
        elif history is not None:
            self.fit(history)
        return self.beta_hat

    def step_forecast(self, state: dict[str, float]) -> dict[str, float]:
        """Advance one deterministic SEIR forecast step."""
        population = state["S"] + state["E"] + state["I"] + state["R"]
        if population <= 0:
            raise ValueError("forecast state population must be positive")
        infection_prob = 1.0 - np.exp(-self.beta_hat * state["I"] / population)
        infection_prob = float(np.clip(infection_prob, 0.0, 1.0))

        new_exposed = state["S"] * infection_prob
        new_infectious = state["E"] * self.sigma_hat
        new_recovered = state["I"] * self.gamma_hat

        return {
            "S": state["S"] - new_exposed,
            "E": state["E"] + new_exposed - new_infectious,
            "I": state["I"] + new_infectious - new_recovered,
            "R": state["R"] + new_recovered,
            "t": state["t"] + 1,
        }

    def forecast(self, current_state: dict[str, float], horizon: int) -> list[dict[str, float]]:
        """Forecast a list of states for the requested horizon."""
        state = deepcopy(current_state)
        forecasts = []
        for _ in range(horizon):
            state = self.step_forecast(state)
            forecasts.append(deepcopy(state))
        return forecasts

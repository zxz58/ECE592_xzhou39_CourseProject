"""Stochastic SEIR simulator used as the hidden data-generating process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SEIRState:
    """Population counts at one time step."""

    S: int
    E: int
    I: int
    R: int
    t: int

    def as_dict(self) -> dict[str, int]:
        return {"S": self.S, "E": self.E, "I": self.I, "R": self.R, "t": self.t}


class StochasticSEIRSimulator:
    """Discrete-time stochastic SEIR simulator with a mid-episode beta surge."""

    def __init__(
        self,
        population: int = 100_000,
        initial_exposed: int = 20,
        initial_infectious: int = 10,
        beta_base: float = 0.30,
        beta_surge: float = 0.65,
        sigma: float = 1 / 5.2,
        gamma: float = 1 / 7,
        episode_length: int = 120,
        surge_start: int = 45,
        surge_end: int = 75,
        beta_schedule: list[float] | np.ndarray | None = None,
    ) -> None:
        self.population = int(population)
        self.initial_exposed = int(initial_exposed)
        self.initial_infectious = int(initial_infectious)
        self.beta_base = float(beta_base)
        self.beta_surge = float(beta_surge)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.episode_length = int(episode_length)
        self.surge_start = int(surge_start)
        self.surge_end = int(surge_end)
        self.beta_schedule = None if beta_schedule is None else np.asarray(beta_schedule, dtype=float)
        self._validate_parameters()
        self.rng = np.random.default_rng()
        self.state = self._initial_state()

    def _validate_parameters(self) -> None:
        """Fail fast for parameters that could create invalid populations."""
        if self.population <= 0:
            raise ValueError("population must be positive")
        if self.initial_exposed < 0 or self.initial_infectious < 0:
            raise ValueError("initial exposed/infectious counts must be nonnegative")
        if self.initial_exposed + self.initial_infectious > self.population:
            raise ValueError("initial exposed + infectious cannot exceed population")
        for name, value in {"sigma": self.sigma, "gamma": self.gamma}.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.beta_base < 0.0 or self.beta_surge < 0.0:
            raise ValueError("beta values must be nonnegative")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.surge_start < 0 or self.surge_end < self.surge_start:
            raise ValueError("surge_start/surge_end must define a valid interval")
        if self.beta_schedule is not None:
            if self.beta_schedule.ndim != 1 or len(self.beta_schedule) != self.episode_length:
                raise ValueError("beta_schedule must contain exactly episode_length values")
            if not np.all(np.isfinite(self.beta_schedule)) or np.any(self.beta_schedule < 0.0):
                raise ValueError("beta_schedule values must be finite and nonnegative")

    def _initial_state(self) -> SEIRState:
        removed = self.initial_exposed + self.initial_infectious
        return SEIRState(
            S=self.population - removed,
            E=self.initial_exposed,
            I=self.initial_infectious,
            R=0,
            t=0,
        )

    def reset(self, seed: int | None = None) -> dict[str, int]:
        """Reset the simulator and optionally reseed its random generator."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self._initial_state()
        return self.get_state()

    def get_true_beta(self) -> float:
        """Return beta for the current simulator time."""
        if self.beta_schedule is not None:
            index = min(self.state.t, self.episode_length - 1)
            return float(self.beta_schedule[index])
        if self.surge_start <= self.state.t <= self.surge_end:
            return self.beta_surge
        return self.beta_base

    def get_beta_schedule(self) -> np.ndarray:
        """Return beta values for the whole episode."""
        if self.beta_schedule is not None:
            return self.beta_schedule.copy()
        return np.array(
            [
                self.beta_surge if self.surge_start <= t <= self.surge_end else self.beta_base
                for t in range(self.episode_length)
            ],
            dtype=float,
        )

    def get_state(self) -> dict[str, int]:
        """Return a copy of the current state as a dictionary."""
        return self.state.as_dict()

    def clone_state(self) -> dict[str, int]:
        """Alias for diagnostics that need an immutable snapshot."""
        return self.get_state().copy()

    def _checked_state(self, S: int, E: int, I: int, R: int, t: int) -> SEIRState:
        """Create a state only after nonnegativity and conservation checks."""
        values = {"S": S, "E": E, "I": I, "R": R}
        if any(value < 0 for value in values.values()):
            raise RuntimeError(f"SEIR compartment went negative: {values}")
        total = S + E + I + R
        if total != self.population:
            raise RuntimeError(f"SEIR population not conserved: {total} != {self.population}")
        return SEIRState(S=S, E=E, I=I, R=R, t=t)

    def step(self) -> dict[str, int]:
        """Advance one day using stochastic SEIR transitions."""
        beta_t = self.get_true_beta()
        state = self.state
        infection_prob = 1.0 - np.exp(-beta_t * state.I / self.population)
        infection_prob = float(np.clip(infection_prob, 0.0, 1.0))

        new_exposed = self.rng.binomial(state.S, infection_prob)
        new_infectious = self.rng.binomial(state.E, self.sigma)
        new_recovered = self.rng.binomial(state.I, self.gamma)

        self.state = self._checked_state(
            S=int(state.S - new_exposed),
            E=int(state.E + new_exposed - new_infectious),
            I=int(state.I + new_infectious - new_recovered),
            R=int(state.R + new_recovered),
            t=state.t + 1,
        )
        return self.get_state()

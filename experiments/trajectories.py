"""Reproducible beta trajectories for the two course-project tiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BetaEvent:
    """One scheduled beta change, inclusive of both start and end days."""

    event: int
    start: int
    duration: int
    end: int
    start_beta: float
    end_beta: float
    duration_group: str

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class BetaTrajectory:
    """A daily beta schedule and the events used to construct it."""

    tier: str
    seed: int | None
    schedule: np.ndarray
    events: tuple[BetaEvent, ...]


def fixed_trajectory() -> BetaTrajectory:
    """Return the exact 160-day trajectory used by the fixed tier."""
    schedule = np.empty(160, dtype=float)
    schedule[0:20] = 0.30
    schedule[20:50] = np.linspace(0.30, 0.60, 30)
    schedule[50:65] = 0.60
    schedule[65:85] = 0.25
    schedule[85:105] = 0.70
    schedule[105:135] = np.linspace(0.70, 0.35, 30)
    schedule[135:160] = 0.35
    events = (
        BetaEvent(1, 20, 30, 49, 0.30, 0.60, "long"),
        BetaEvent(2, 65, 1, 65, 0.60, 0.25, "short"),
        BetaEvent(3, 85, 1, 85, 0.25, 0.70, "short"),
        BetaEvent(4, 105, 30, 134, 0.70, 0.35, "long"),
    )
    return BetaTrajectory("fixed", None, schedule, events)


def _truncated_rounded_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lower: int,
    upper: int,
) -> int:
    """Sample a normal variate truncated before rounding to integer days."""
    while True:
        sample = float(rng.normal(mean, std))
        if lower <= sample <= upper:
            return int(np.clip(np.rint(sample), lower, upper))


def _sample_duration(rng: np.random.Generator) -> tuple[int, str]:
    if rng.random() < 0.5:
        return _truncated_rounded_normal(rng, 3.0, 1.22, 1, 10), "short"
    return _truncated_rounded_normal(rng, 25.0, 7.4, 11, 40), "long"


def _sample_target_beta(rng: np.random.Generator, current_beta: float) -> float:
    while True:
        target = float(rng.uniform(0.15, 0.85))
        if abs(target - current_beta) >= 0.10:
            return target


def random_trajectory(seed: int, length: int = 240) -> BetaTrajectory:
    """Generate four random beta changes that finish inside the episode cap."""
    if length <= 0:
        raise ValueError("length must be positive")
    rng = np.random.default_rng(seed)
    while True:
        events: list[BetaEvent] = []
        previous_end = -1
        current_beta = 0.30
        for event_index in range(1, 5):
            gap = int(rng.integers(2, 31))
            start = previous_end + 1 + gap
            duration, duration_group = _sample_duration(rng)
            end = start + duration - 1
            target_beta = _sample_target_beta(rng, current_beta)
            events.append(
                BetaEvent(
                    event=event_index,
                    start=start,
                    duration=duration,
                    end=end,
                    start_beta=current_beta,
                    end_beta=target_beta,
                    duration_group=duration_group,
                )
            )
            previous_end = end
            current_beta = target_beta
        if events[-1].end < length:
            break

    schedule = np.full(length, 0.30, dtype=float)
    current_beta = 0.30
    cursor = 0
    for event in events:
        schedule[cursor : event.start] = current_beta
        if event.duration == 1:
            schedule[event.start] = event.end_beta
        else:
            schedule[event.start : event.end + 1] = np.linspace(
                event.start_beta, event.end_beta, event.duration
            )
        current_beta = event.end_beta
        cursor = event.end + 1
    schedule[cursor:] = current_beta
    return BetaTrajectory("random", int(seed), schedule, tuple(events))


def manifest_rows(
    trajectory: BetaTrajectory,
    split: str,
    episode: int,
    epidemic_seed: int,
) -> list[dict[str, int | float | str | None]]:
    """Return one serializable manifest row per trajectory event."""
    return [
        {
            "tier": trajectory.tier,
            "split": split,
            "episode": int(episode),
            "epidemic_seed": int(epidemic_seed),
            "trajectory_seed": trajectory.seed,
            **event.as_dict(),
        }
        for event in trajectory.events
    ]

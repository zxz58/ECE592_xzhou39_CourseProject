"""Benchmark beta recalibration methods on a fixed multi-regime timeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnostics.plots import plot_beta_timeline_benchmark
from envs.recalibration_env import RECALIBRATE, WAIT
from experiments.common import make_env, save_csv
from utils.config import load_config


DEFAULT_CONFIG = "configs/beta_recalibration_timeline.yaml"


def build_beta_schedule(config: dict[str, Any]) -> np.ndarray:
    """Build and validate an explicit beta schedule from configured segments."""
    episode_length = int(config["simulator"]["episode_length"])
    schedule = np.full(episode_length, np.nan, dtype=float)
    segments = config["benchmark"]["beta_segments"]
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if start < 0 or end < start or end >= episode_length:
            raise ValueError(f"Invalid beta segment bounds: {start}..{end}")
        if np.any(np.isfinite(schedule[start : end + 1])):
            raise ValueError(f"Overlapping beta segment: {start}..{end}")
        schedule[start : end + 1] = np.linspace(
            float(segment["start_beta"]),
            float(segment["end_beta"]),
            end - start + 1,
        )
    if np.any(~np.isfinite(schedule)):
        missing = np.flatnonzero(~np.isfinite(schedule)).tolist()
        raise ValueError(f"Beta segments do not cover episode days: {missing}")
    if np.any(schedule < 0.0):
        raise ValueError("Beta schedule values must be nonnegative")
    return schedule


def build_trigger_metadata(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return event metadata keyed by the simulator time of each trigger."""
    benchmark = config["benchmark"]
    episode_length = int(config["simulator"]["episode_length"])
    metadata: dict[int, dict[str, Any]] = {}
    jump_days = {str(key): int(value) for key, value in benchmark.get("jump_days", {}).items()}
    for event_type, days in benchmark["trigger_groups"].items():
        for day_value in days:
            day = int(day_value)
            if day < 0 or day >= episode_length:
                raise ValueError(f"Trigger day {day} is outside the episode")
            if day in metadata:
                raise ValueError(f"Duplicate trigger day: {day}")
            event_offset = day - jump_days[event_type] if event_type in jump_days else np.nan
            metadata[day] = {"event_type": str(event_type), "event_offset": event_offset}
    return metadata


def regime_metadata(config: dict[str, Any], day: int) -> tuple[str, str]:
    """Return the named regime and broad phase type for a day."""
    for segment in config["benchmark"]["beta_segments"]:
        if int(segment["start"]) <= day <= int(segment["end"]):
            return str(segment["name"]), str(segment["phase_type"])
    raise ValueError(f"No beta regime configured for day {day}")


def configure_method(config: dict[str, Any], method: str, fit_window: int) -> dict[str, Any]:
    """Return an isolated environment configuration for one method."""
    method_config = deepcopy(config)
    method_config["recalibration"] = dict(method_config["recalibration"])
    method_config["recalibration"]["method"] = method
    method_config["recalibration"]["fit_window"] = int(fit_window)
    method_config["surrogate"] = dict(method_config["surrogate"])
    method_config["surrogate"]["fit_window"] = int(fit_window)
    return method_config


def _percentage_error(estimate: float, truth: float) -> float:
    return 100.0 * (estimate - truth) / truth


def run_single_episode(
    config: dict[str, Any],
    method: str,
    seed: int,
    repetition: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one method on one paired epidemic seed."""
    env = make_env(config)
    env.reset(seed=seed)
    triggers = build_trigger_metadata(config)
    step_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []

    done = False
    while not done:
        current_state = env.true_simulator.get_state()
        day = int(current_state["t"])
        true_beta_at_step_start = float(env.true_simulator.get_true_beta())
        regime, phase_type = regime_metadata(config, day)
        trigger = triggers.get(day)
        action = RECALIBRATE if trigger is not None else WAIT
        active_at_start = float(current_state["E"] + current_state["I"])

        result = env.step(action)
        info = result.info
        forecast_path = info["surrogate_forecast"]
        forecast_target_day_7d = day + 7
        has_7d_target = forecast_target_day_7d <= env.horizon
        step_rows.append(
            {
                "method": method,
                "repetition": int(repetition),
                "seed": int(seed),
                "day": day,
                "next_day": int(info["t"]),
                "regime": regime,
                "phase_type": phase_type,
                "event_type": trigger["event_type"] if trigger else "none",
                "triggered": int(trigger is not None),
                "true_beta": true_beta_at_step_start,
                "beta_hat": float(info["beta_hat"]),
                "forecast_error": float(info["forecast_error"]),
                "cumulative_error": float(info["cumulative_error"]),
                "active_infections_at_start": active_at_start,
                "true_S": float(info["true_state"]["S"]),
                "true_E": float(info["true_state"]["E"]),
                "true_I": float(info["true_state"]["I"]),
                "surrogate_E": float(forecast_path[0]["E"]),
                "forecast_target_day_7d": forecast_target_day_7d if has_7d_target else np.nan,
                "surrogate_I_7d": float(forecast_path[6]["I"]) if has_7d_target else np.nan,
                "true_R": float(info["true_state"]["R"]),
            }
        )

        if trigger is not None:
            applied_beta = float(info["recalibration_applied_beta"])
            skipped = bool(info["recalibration_skipped"])
            selected_beta = float(info["recalibration_selected_beta"])
            signed_error = _percentage_error(applied_beta, true_beta_at_step_start)
            selected_signed_error = (
                np.nan if skipped else _percentage_error(selected_beta, true_beta_at_step_start)
            )
            trigger_rows.append(
                {
                    "method": method,
                    "repetition": int(repetition),
                    "seed": int(seed),
                    "trigger_day": day,
                    "regime": regime,
                    "phase_type": phase_type,
                    "event_type": trigger["event_type"],
                    "event_offset": trigger["event_offset"],
                    "true_beta_at_trigger": true_beta_at_step_start,
                    "old_beta": float(info["recalibration_old_beta"]),
                    "selected_beta": selected_beta,
                    "applied_beta": applied_beta,
                    "signed_error_pct": signed_error,
                    "absolute_error_pct": abs(signed_error),
                    "selected_signed_error_pct": selected_signed_error,
                    "selected_absolute_error_pct": abs(selected_signed_error),
                    "fit_loss": float(info["recalibration_fit_loss"]),
                    "valid_pairs": int(info["recalibration_valid_pairs"]),
                    "active_infections": active_at_start,
                    "skipped": int(skipped),
                }
            )
        done = result.done

    return pd.DataFrame(step_rows), pd.DataFrame(trigger_rows)


def _aggregate_trigger_accuracy(
    trigger_results: pd.DataFrame,
    group_columns: list[str],
    extra_aggregations: dict[str, tuple[str, Any]],
) -> pd.DataFrame:
    """Apply the shared beta-accuracy metrics to a trigger grouping."""
    aggregations = {
        "repetitions": ("seed", "nunique"),
        "bias_pct": ("signed_error_pct", "mean"),
        "mae_pct": ("absolute_error_pct", "mean"),
        "error_std_pct": ("signed_error_pct", "std"),
        "median_error_pct": ("signed_error_pct", "median"),
        "median_absolute_error_pct": ("absolute_error_pct", "median"),
        "selected_bias_pct": ("selected_signed_error_pct", "mean"),
        "selected_mae_pct": ("selected_absolute_error_pct", "mean"),
        "skip_rate": ("skipped", "mean"),
        "valid_pairs_mean": ("valid_pairs", "mean"),
        **extra_aggregations,
    }
    summary = trigger_results.groupby(group_columns, as_index=False, dropna=False).agg(
        **aggregations
    )
    squared = (
        trigger_results.assign(squared_error=trigger_results["signed_error_pct"] ** 2)
        .groupby(group_columns, as_index=False, dropna=False)["squared_error"]
        .mean()
        .rename(columns={"squared_error": "mean_squared_error"})
    )
    summary = summary.merge(squared, on=group_columns, how="left")
    summary["rmse_pct"] = np.sqrt(summary.pop("mean_squared_error"))
    return summary


def summarize_triggers(trigger_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate estimator accuracy at each configured trigger."""
    group_columns = ["method", "trigger_day", "regime", "phase_type", "event_type", "event_offset"]
    summary = _aggregate_trigger_accuracy(
        trigger_results,
        group_columns,
        {
            "error_p10_pct": ("signed_error_pct", lambda values: values.quantile(0.10)),
            "error_p90_pct": ("signed_error_pct", lambda values: values.quantile(0.90)),
            "active_infections_mean": ("active_infections", "mean"),
        },
    )
    return summary.sort_values(["method", "trigger_day"]).reset_index(drop=True)


def summarize_trigger_groups(trigger_results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate applied-beta accuracy over configured events or regimes."""
    summary = _aggregate_trigger_accuracy(
        trigger_results,
        group_columns,
        {"triggers": ("trigger_day", "nunique")},
    )
    return summary.sort_values(group_columns).reset_index(drop=True)


def run(
    config_path: str = DEFAULT_CONFIG,
    repetitions: int | None = None,
    base_seed: int | None = None,
    fit_window: int | None = None,
    methods: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the paired fixed-timeline benchmark and save tables and figures."""
    config = load_config(config_path)
    schedule = build_beta_schedule(config)
    config["simulator"] = dict(config["simulator"])
    config["simulator"]["beta_schedule"] = schedule.tolist()

    benchmark = config["benchmark"]
    repetitions = int(benchmark["repetitions"] if repetitions is None else repetitions)
    base_seed = int(config["seed"] if base_seed is None else base_seed)
    fit_window = int(config["recalibration"]["fit_window"] if fit_window is None else fit_window)
    methods = list(benchmark["methods"] if methods is None else methods)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if fit_window <= 0:
        raise ValueError("fit_window must be positive")

    root = Path(config["outputs"]["results_dir"] if output_dir is None else output_dir)
    figures_dir = root / "figures"
    step_frames: list[pd.DataFrame] = []
    trigger_frames: list[pd.DataFrame] = []

    for method in methods:
        method_config = configure_method(config, method, fit_window)
        for repetition in range(repetitions):
            seed = base_seed + repetition
            steps, triggers = run_single_episode(method_config, method, seed, repetition)
            step_frames.append(steps)
            trigger_frames.append(triggers)

    step_trajectories = pd.concat(step_frames, ignore_index=True)
    trigger_results = pd.concat(trigger_frames, ignore_index=True)
    trigger_summary = summarize_triggers(trigger_results)
    trigger_event_summary = summarize_trigger_groups(trigger_results, ["method", "event_type"])
    trigger_regime_summary = summarize_trigger_groups(
        trigger_results, ["method", "regime", "phase_type"]
    )
    outputs = {
        "step_trajectories": step_trajectories,
        "trigger_results": trigger_results,
        "trigger_summary": trigger_summary,
        "trigger_event_summary": trigger_event_summary,
        "trigger_regime_summary": trigger_regime_summary,
    }
    for name, frame in outputs.items():
        save_csv(frame, root / f"beta_timeline_{name}.csv")
    plot_beta_timeline_benchmark(
        step_trajectories=step_trajectories,
        trigger_results=trigger_results,
        figures_dir=figures_dir,
        method_order=methods,
        jump_days={str(key): int(value) for key, value in benchmark.get("jump_days", {}).items()},
    )

    print(trigger_summary.to_string(index=False))
    print(f"saved benchmark results to: {root}")
    print(f"saved benchmark figures to: {figures_dir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--fit-window", type=int)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(
        config_path=args.config,
        repetitions=args.repetitions,
        base_seed=args.base_seed,
        fit_window=args.fit_window,
        methods=args.methods,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

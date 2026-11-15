"""Compare compact error-state designs for DQN recalibration."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from diagnostics.metrics import summarize_trajectories
from diagnostics.plots import plot_error_feature_ablation
from experiments.common import evaluate_policy, make_env, save_csv
from experiments.run_error_history_sweep import DQNPolicy, _train_agent
from utils.config import ensure_output_dirs, load_config


STATE_DESIGNS = {
    "A_raw_errors": [
        "normalized_compartments",
        "recent_errors",
        "remaining_budget",
        "days_since_recalibration",
    ],
    "B_last_mean": [
        "normalized_compartments",
        "last_forecast_error",
        "rolling_mean_forecast_error",
        "remaining_budget",
        "days_since_recalibration",
    ],
    "C_last_mean_trend": [
        "normalized_compartments",
        "last_forecast_error",
        "rolling_mean_forecast_error",
        "rolling_error_trend",
        "remaining_budget",
        "days_since_recalibration",
    ],
    "D_raw_mean_trend": [
        "normalized_compartments",
        "recent_errors",
        "rolling_mean_forecast_error",
        "rolling_error_trend",
        "remaining_budget",
        "days_since_recalibration",
    ],
}


def _candidate_pairs(history_lengths: list[int], candidates: list[str] | None) -> list[tuple[str, int]]:
    """Return explicit design/K pairs, or the full design x K grid."""
    if not candidates:
        return [
            (design_name, history_length)
            for design_name in STATE_DESIGNS
            for history_length in history_lengths
        ]

    pairs = []
    for candidate in candidates:
        if ":" not in candidate:
            raise ValueError("Candidates must use the form state_design:history_length")
        design_name, history_length = candidate.split(":", 1)
        if design_name not in STATE_DESIGNS:
            raise ValueError(f"Unknown state design: {design_name}")
        pairs.append((design_name, int(history_length)))
    return pairs


def run(
    config_path: str,
    history_lengths: list[int],
    seeds: list[int],
    episodes: int,
    candidates: list[str] | None = None,
) -> None:
    """Train and evaluate each error-state design."""
    base_config = load_config(config_path)
    base_config["dqn"]["episodes"] = episodes
    ensure_output_dirs(base_config)
    results_dir = Path(base_config["outputs"]["results_dir"])
    figures_dir = Path(base_config["outputs"].get("figures_dir", results_dir / "figures"))
    ablation_root = results_dir / "error_feature_ablation"
    detail_rows = []

    for design_name, history_length in _candidate_pairs(history_lengths, candidates):
        state_features = STATE_DESIGNS[design_name]
        for seed in seeds:
            config = deepcopy(base_config)
            config["seed"] = seed
            config["environment"]["error_history_length"] = history_length
            config["environment"]["state_features"] = state_features
            run_dir = ablation_root / design_name / f"k_{history_length}" / f"seed_{seed}"
            print(f"training {design_name}, K={history_length}, seed={seed}")
            agent, _ = _train_agent(config, seed, run_dir)

            env = make_env(config)
            trajectories = evaluate_policy(
                env=env,
                policy=DQNPolicy(agent),
                episodes=config["evaluation"]["episodes"],
                seed=seed + 10_000,
            )
            save_csv(trajectories, run_dir / "results" / "dqn_eval_trajectories.csv")
            episode_summary = summarize_trajectories(trajectories)
            save_csv(episode_summary, run_dir / "results" / "dqn_eval_episode_summary.csv")

            detail_rows.append(
                {
                    "state_design": design_name,
                    "error_history_length": history_length,
                    "training_seed": seed,
                    "observation_size": make_env(config).observation_size,
                    "cumulative_error_mean": episode_summary["cumulative_error"].mean(),
                    "cumulative_error_std": episode_summary["cumulative_error"].std(),
                    "average_forecast_error_mean": episode_summary["average_forecast_error"].mean(),
                    "budget_used_mean": episode_summary["budget_used"].mean(),
                    "episode_length_mean": episode_summary["episode_length"].mean(),
                }
            )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["state_design", "error_history_length"], as_index=False)
        .agg(
            cumulative_error_mean=("cumulative_error_mean", "mean"),
            cumulative_error_std=("cumulative_error_mean", "std"),
            cumulative_error_median=("cumulative_error_mean", "median"),
            cumulative_error_worst_seed=("cumulative_error_mean", "max"),
            average_forecast_error_mean=("average_forecast_error_mean", "mean"),
            budget_used_mean=("budget_used_mean", "mean"),
            episode_length_mean=("episode_length_mean", "mean"),
            observation_size=("observation_size", "first"),
            n_training_seeds=("training_seed", "nunique"),
        )
        .sort_values(["state_design", "error_history_length"])
    )

    save_csv(detail, results_dir / "error_feature_ablation_detailed.csv")
    save_csv(summary, results_dir / "error_feature_ablation_summary.csv")
    plot_error_feature_ablation(summary, figures_dir / "error_feature_ablation_cumulative_error.png")
    print(summary.to_string(index=False))
    print(f"saved detailed results: {results_dir / 'error_feature_ablation_detailed.csv'}")
    print(f"saved summary results: {results_dir / 'error_feature_ablation_summary.csv'}")
    print(f"saved figure: {figures_dir / 'error_feature_ablation_cumulative_error.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--history-lengths", nargs="+", type=int, default=[7, 11])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument(
        "--candidates",
        nargs="+",
        help="Optional explicit pairs like A_raw_errors:10 D_raw_mean_trend:15",
    )
    args = parser.parse_args()
    run(args.config, args.history_lengths, args.seeds, args.episodes, args.candidates)


if __name__ == "__main__":
    main()

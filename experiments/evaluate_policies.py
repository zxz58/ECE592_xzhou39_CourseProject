"""Evaluate fixed schedule, threshold, and DQN recalibration policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.dqn_agent import DQNAgent
from baselines.fixed_schedule import FixedSchedulePolicy
from baselines.threshold import ThresholdPolicy
from diagnostics.metrics import compare_policies, summarize_trajectories
from diagnostics.plots import (
    plot_beta_tracking,
    plot_average_daily_error_comparison,
    plot_budget_usage_comparison,
    plot_cumulative_error_comparison,
    plot_episode_length_distribution,
    plot_episode_diagnostics,
    plot_forecast_error,
    plot_normalized_recalibration_timing,
    plot_policy_errors,
    plot_representative_episode_diagnostics,
    plot_true_seir_curves,
)
from experiments.common import (
    evaluate_policy,
    make_env,
    representative_fixed_interval,
    representative_threshold,
    save_csv,
)
from utils.config import ensure_output_dirs, load_config
from utils.seed import set_seed


class DQNPolicy:
    """Evaluation wrapper around a trained DQN agent."""

    name = "dqn"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def reset(self) -> None:
        """No recurrent state is used."""

    def action(self, state, info: dict) -> int:
        """Select the greedy DQN action for evaluation."""
        return self.agent.select_action(state, training=False)


def _safe_name(name: str) -> str:
    """Make policy names filesystem-friendly."""
    return name.replace(".", "p").replace("/", "_")


def build_policies(config: dict, model_path: str | Path) -> list:
    """Create fixed schedule, threshold, and DQN policies."""
    policies = [
        FixedSchedulePolicy(interval=interval)
        for interval in config["evaluation"].get("fixed_intervals", [7, 14, 21, 28])
    ]
    policies.extend(
        ThresholdPolicy(
            error_threshold=threshold,
            error_history_length=config["environment"].get("error_history_length", 7),
        )
        for threshold in config["evaluation"].get("thresholds", [0.001, 0.002, 0.003, 0.005])
    )

    env = make_env(config)
    agent = DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        hidden_dim=config["dqn"]["hidden_dim"],
        learning_rate=config["dqn"]["learning_rate"],
        gamma=config["dqn"]["gamma"],
        replay_capacity=config["dqn"]["replay_capacity"],
        batch_size=config["dqn"]["batch_size"],
        epsilon_start=config["dqn"]["epsilon_start"],
        epsilon_end=config["dqn"]["epsilon_end"],
        epsilon_decay_steps=config["dqn"]["epsilon_decay_steps"],
        target_update_interval=config["dqn"]["target_update_interval"],
        min_replay_size=config["dqn"]["min_replay_size"],
        seed=config["seed"],
    )
    agent.load(model_path)
    policies.append(DQNPolicy(agent))
    return policies


def evaluate(config_path: str, model_path: str) -> None:
    """Run policy-grid evaluation and save summaries plus episode logs."""
    config = load_config(config_path)
    ensure_output_dirs(config)
    set_seed(config["seed"])

    results_dir = Path(config["outputs"]["results_dir"])
    figures_dir = Path(config["outputs"].get("figures_dir", results_dir / "figures"))
    episode_log_dir = results_dir / "episode_logs"
    episode_log_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for offset, policy in enumerate(build_policies(config, model_path)):
        env = make_env(config)
        trajectories = evaluate_policy(
            env=env,
            policy=policy,
            episodes=config["evaluation"]["episodes"],
            seed=config["seed"] + 10_000 * offset,
        )
        frames.append(trajectories)
        save_csv(trajectories, episode_log_dir / f"{_safe_name(policy.name)}.csv")

    all_trajectories = pd.concat(frames, ignore_index=True)
    episode_summary = summarize_trajectories(all_trajectories)
    evaluation_summary = compare_policies(episode_summary)

    save_csv(evaluation_summary, results_dir / "evaluation_summary.csv")
    save_csv(episode_summary, results_dir / "policy_episode_summary.csv")
    save_csv(all_trajectories, results_dir / "policy_trajectories.csv")
    # Keep the previous filename as a convenience for older notebooks.
    save_csv(evaluation_summary, results_dir / "policy_comparison.csv")

    plot_policy_errors(episode_summary, figures_dir / "policy_error_comparison.png")
    plot_cumulative_error_comparison(episode_summary, figures_dir / "cumulative_error_by_policy.png")
    plot_budget_usage_comparison(episode_summary, figures_dir / "budget_usage_by_policy.png")
    plot_episode_length_distribution(episode_summary, figures_dir / "episode_length_distribution.png")
    plot_average_daily_error_comparison(episode_summary, figures_dir / "average_daily_error_by_policy.png")
    plot_normalized_recalibration_timing(
        episode_summary,
        figures_dir / "normalized_recalibration_timing.png",
        surge_start=config["simulator"].get("surge_start"),
        surge_end=config["simulator"].get("surge_end"),
    )
    plot_episode_diagnostics(all_trajectories, figures_dir / "episode_diagnostics.png", episode=0)
    fixed_interval = representative_fixed_interval(config)
    threshold = representative_threshold(config)
    representative_policies = [
        "dqn",
        f"threshold_{threshold}",
        f"fixed_schedule_{fixed_interval}",
    ]
    plot_representative_episode_diagnostics(
        all_trajectories,
        figures_dir / "representative_episode_diagnostics.png",
        episode=0,
        policies=representative_policies,
        surge_start=config["simulator"].get("surge_start"),
        surge_end=config["simulator"].get("surge_end"),
    )

    diagnostic_policy = "dqn" if "dqn" in set(all_trajectories["policy"]) else all_trajectories["policy"].iloc[0]
    plot_true_seir_curves(
        all_trajectories,
        figures_dir / "true_seir_curves.png",
        episode=0,
        policy=diagnostic_policy,
    )
    plot_beta_tracking(
        all_trajectories,
        figures_dir / "true_beta_vs_surrogate_beta.png",
        episode=0,
        policy=diagnostic_policy,
    )
    plot_forecast_error(
        all_trajectories,
        figures_dir / "forecast_error_over_time.png",
        episode=0,
        policy=diagnostic_policy,
    )

    print(evaluation_summary.to_string(index=False))
    print(f"saved evaluation summary: {results_dir / 'evaluation_summary.csv'}")
    print(f"saved episode logs to: {episode_log_dir}")
    print(f"saved figures to: {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="results/dqn_model.pt")
    args = parser.parse_args()
    evaluate(args.config, args.model)


if __name__ == "__main__":
    main()

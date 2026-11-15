"""Run robustness checks across simulator scenarios."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from agents.dqn_agent import DQNAgent
from baselines.fixed_schedule import FixedSchedulePolicy
from baselines.threshold import ThresholdPolicy
from diagnostics.metrics import compare_policies, summarize_trajectories
from diagnostics.plots import plot_robustness_cumulative_error
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
    """Greedy DQN evaluation policy."""

    name = "dqn"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def reset(self) -> None:
        """No recurrent state."""

    def action(self, state, info: dict) -> int:
        return self.agent.select_action(state, training=False)


def scenario_configs(base_config: dict) -> list[tuple[str, dict]]:
    """Create simulator scenario variants for robustness checks."""
    scenarios = []
    definitions = [
        ("baseline", {}),
        ("low_beta_base", {"beta_base": 0.22}),
        ("high_beta_base", {"beta_base": 0.38}),
        ("early_surge", {"surge_start": 30, "surge_end": 60}),
        ("late_surge", {"surge_start": 60, "surge_end": 90}),
        ("mild_surge", {"beta_surge": 0.50}),
        ("severe_surge", {"beta_surge": 0.85}),
        ("low_initial_infections", {"initial_exposed": 10, "initial_infectious": 5}),
        ("high_initial_infections", {"initial_exposed": 60, "initial_infectious": 30}),
    ]
    for name, overrides in definitions:
        config = deepcopy(base_config)
        config["simulator"].update(overrides)
        scenarios.append((name, config))
    return scenarios


def build_dqn_policy(config: dict, model_path: str | Path) -> DQNPolicy:
    """Load a trained DQN policy."""
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
    return DQNPolicy(agent)


def run(config_path: str, model_path: str, seeds: list[int]) -> None:
    """Evaluate policy robustness across scenarios and seeds."""
    base_config = load_config(config_path)
    ensure_output_dirs(base_config)
    set_seed(base_config["seed"])

    frames = []
    for scenario_name, config in scenario_configs(base_config):
        dqn_policy = build_dqn_policy(config, model_path)
        fixed_interval = representative_fixed_interval(config)
        threshold = representative_threshold(config)
        policies = [
            FixedSchedulePolicy(interval=fixed_interval),
            ThresholdPolicy(
                error_threshold=threshold,
                error_history_length=config["environment"].get("error_history_length", 7),
            ),
            dqn_policy,
        ]
        for seed in seeds:
            for offset, policy in enumerate(policies):
                env = make_env(config)
                data = evaluate_policy(
                    env,
                    policy,
                    episodes=1,
                    seed=seed + 10_000 * offset,
                )
                data["scenario"] = scenario_name
                data["evaluation_seed"] = seed
                frames.append(data)

    trajectories = pd.concat(frames, ignore_index=True)
    rows = []
    for (scenario, evaluation_seed, policy), group in trajectories.groupby(
        ["scenario", "evaluation_seed", "policy"]
    ):
        row = summarize_trajectories(group).iloc[0].to_dict()
        row["scenario"] = scenario
        row["evaluation_seed"] = evaluation_seed
        row["policy"] = policy
        rows.append(row)
    episode_summary = pd.DataFrame(rows)
    comparison = (
        episode_summary.groupby(["scenario", "policy"], as_index=False)
        .agg(
            cumulative_error_mean=("cumulative_error", "mean"),
            cumulative_error_std=("cumulative_error", "std"),
            average_forecast_error_mean=("average_forecast_error", "mean"),
            average_forecast_error_std=("average_forecast_error", "std"),
            episode_length_mean=("episode_length", "mean"),
            budget_used_mean=("budget_used", "mean"),
            budget_utilization_mean=("budget_utilization", "mean"),
            final_epidemic_size_mean=("final_epidemic_size", "mean"),
        )
    )

    results_dir = Path(base_config["outputs"]["results_dir"])
    figures_dir = Path(base_config["outputs"].get("figures_dir", results_dir / "figures"))
    save_csv(trajectories, results_dir / "robustness_trajectories.csv")
    save_csv(episode_summary, results_dir / "robustness_summary.csv")
    save_csv(comparison, results_dir / "robustness_comparison.csv")
    plot_robustness_cumulative_error(
        episode_summary,
        figures_dir / "robustness_cumulative_error.png",
    )
    print(f"saved robustness summary table: {results_dir / 'robustness_comparison.csv'}")
    print(f"saved robustness figures to: {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="results/dqn_model.pt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    args = parser.parse_args()
    run(args.config, args.model, args.seeds)


if __name__ == "__main__":
    main()

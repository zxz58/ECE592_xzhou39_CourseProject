"""Compare policies across recalibration budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.dqn_agent import DQNAgent
from baselines.fixed_schedule import FixedSchedulePolicy
from baselines.threshold import ThresholdPolicy
from diagnostics.metrics import summarize_trajectories
from diagnostics.plots import plot_budget_sweep
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


def run(config_path: str, model_path: str, budgets: list[int]) -> None:
    """Run budget sweep and save CSV/PNG outputs."""
    config = load_config(config_path)
    ensure_output_dirs(config)
    set_seed(config["seed"])

    frames = []
    dqn_policy = build_dqn_policy(config, model_path)
    fixed_interval = representative_fixed_interval(config)
    threshold = representative_threshold(config)
    for budget in budgets:
        policies = [
            FixedSchedulePolicy(interval=fixed_interval),
            ThresholdPolicy(
                error_threshold=threshold,
                error_history_length=config["environment"].get("error_history_length", 7),
            ),
            dqn_policy,
        ]
        for offset, policy in enumerate(policies):
            env = make_env(config, budget=budget)
            data = evaluate_policy(
                env,
                policy,
                episodes=config["evaluation"]["episodes"],
                seed=config["seed"] + 1_000 * budget + offset,
            )
            data["budget"] = budget
            frames.append(data)

    trajectories = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for budget, group in trajectories.groupby("budget"):
        summary = summarize_trajectories(group)
        summary["budget"] = budget
        summary_rows.append(summary)
    episode_summary = pd.concat(summary_rows, ignore_index=True)

    results_dir = Path(config["outputs"]["results_dir"])
    figures_dir = Path(config["outputs"].get("figures_dir", results_dir / "figures"))
    save_csv(trajectories, results_dir / "budget_sweep_trajectories.csv")
    save_csv(episode_summary, results_dir / "budget_sweep_summary.csv")
    plot_budget_sweep(
        episode_summary,
        figures_dir / "budget_sweep_cumulative_error.png",
        figures_dir / "budget_sweep_budget_used.png",
    )
    print(f"saved budget sweep results to: {results_dir}")
    print(f"saved budget sweep figures to: {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="results/dqn_model.pt")
    parser.add_argument("--budgets", nargs="+", type=int, default=[0, 1, 3, 5, 10, 20])
    args = parser.parse_args()
    run(args.config, args.model, args.budgets)


if __name__ == "__main__":
    main()

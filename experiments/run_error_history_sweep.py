"""Compare DQN performance across error-history lengths."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from agents.dqn_agent import DQNAgent
from diagnostics.metrics import summarize_trajectories
from diagnostics.plots import (
    plot_epsilon_curve,
    plot_error_history_sweep,
    plot_training_loss_curve,
    plot_training_reward_curve,
)
from experiments.common import evaluate_policy, make_env, save_csv
from utils.config import ensure_output_dirs, load_config
from utils.seed import set_seed


class DQNPolicy:
    """Greedy wrapper used for DQN-only evaluation."""

    name = "dqn"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def reset(self) -> None:
        """DQN has no recurrent state."""

    def action(self, state, info: dict) -> int:
        """Use greedy actions during evaluation."""
        return self.agent.select_action(state, training=False)


def _make_agent(config: dict, seed: int) -> DQNAgent:
    """Create a DQN agent sized to the configured environment."""
    env = make_env(config)
    dqn_cfg = config["dqn"]
    return DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        hidden_dim=dqn_cfg["hidden_dim"],
        learning_rate=dqn_cfg["learning_rate"],
        gamma=dqn_cfg["gamma"],
        replay_capacity=dqn_cfg["replay_capacity"],
        batch_size=dqn_cfg["batch_size"],
        epsilon_start=dqn_cfg["epsilon_start"],
        epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay_steps=dqn_cfg["epsilon_decay_steps"],
        target_update_interval=dqn_cfg["target_update_interval"],
        min_replay_size=dqn_cfg["min_replay_size"],
        seed=seed,
    )


def _train_agent(config: dict, seed: int, run_dir: Path) -> tuple[DQNAgent, pd.DataFrame]:
    """Train one DQN and save its checkpoint plus training diagnostics."""
    set_seed(seed)
    env = make_env(config)
    agent = _make_agent(config, seed)
    rows = []

    for episode in range(config["dqn"]["episodes"]):
        observation = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        losses = []
        final_info = {}

        while not done:
            action = agent.select_action(observation, training=True)
            next_state, reward, done, info = env.step(action)
            final_info = info
            agent.store_transition(observation, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            observation = next_state
            episode_reward += reward

        rows.append(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "cumulative_reward": episode_reward,
                "cumulative_error": env.cumulative_error,
                "epsilon": agent.current_epsilon(),
                "mean_loss": float(np.mean(losses)) if losses else np.nan,
                "budget_used": env.budget - env.remaining_budget,
                "budget_allowed": env.budget,
                "episode_length": int(final_info.get("t", 0)),
                "termination_reason": final_info.get("termination_reason", ""),
            }
        )

    training_log = pd.DataFrame(rows)
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    agent.save(results_dir / "dqn_model.pt")
    save_csv(training_log, results_dir / "dqn_training_log.csv")
    plot_training_reward_curve(training_log, figures_dir / "dqn_training_reward.png")
    plot_training_loss_curve(training_log, figures_dir / "dqn_training_loss.png")
    plot_epsilon_curve(training_log, figures_dir / "dqn_epsilon.png")
    return agent, training_log


def run(config_path: str, history_lengths: list[int], seeds: list[int]) -> None:
    """Train/evaluate DQN for each K and seed, then save summary outputs."""
    base_config = load_config(config_path)
    ensure_output_dirs(base_config)
    results_dir = Path(base_config["outputs"]["results_dir"])
    figures_dir = Path(base_config["outputs"].get("figures_dir", results_dir / "figures"))
    sweep_root = results_dir / "error_history_sweep"
    detail_rows = []

    for history_length in history_lengths:
        for seed in seeds:
            config = deepcopy(base_config)
            config["seed"] = seed
            config["environment"]["error_history_length"] = history_length
            run_dir = sweep_root / f"k_{history_length}" / f"seed_{seed}"
            print(f"training DQN with K={history_length}, seed={seed}")
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

            row = {
                "error_history_length": history_length,
                "training_seed": seed,
                "observation_size": make_env(config).observation_size,
                "cumulative_error_mean": episode_summary["cumulative_error"].mean(),
                "cumulative_error_std": episode_summary["cumulative_error"].std(),
                "average_forecast_error_mean": episode_summary["average_forecast_error"].mean(),
                "budget_used_mean": episode_summary["budget_used"].mean(),
                "episode_length_mean": episode_summary["episode_length"].mean(),
            }
            detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby("error_history_length", as_index=False)
        .agg(
            cumulative_error_mean=("cumulative_error_mean", "mean"),
            cumulative_error_std=("cumulative_error_mean", "std"),
            average_forecast_error_mean=("average_forecast_error_mean", "mean"),
            budget_used_mean=("budget_used_mean", "mean"),
            episode_length_mean=("episode_length_mean", "mean"),
            observation_size=("observation_size", "first"),
            n_training_seeds=("training_seed", "nunique"),
        )
        .sort_values("error_history_length")
    )

    save_csv(detail, results_dir / "error_history_sweep_detailed.csv")
    save_csv(summary, results_dir / "error_history_sweep_summary.csv")
    plot_error_history_sweep(summary, figures_dir / "error_history_sweep_cumulative_error.png")
    print(summary.to_string(index=False))
    print(f"saved detailed results: {results_dir / 'error_history_sweep_detailed.csv'}")
    print(f"saved summary results: {results_dir / 'error_history_sweep_summary.csv'}")
    print(f"saved figure: {figures_dir / 'error_history_sweep_cumulative_error.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--history-lengths", nargs="+", type=int, default=[1, 3, 5, 7, 9, 11, 13])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    args = parser.parse_args()
    run(args.config, args.history_lengths, args.seeds)


if __name__ == "__main__":
    main()

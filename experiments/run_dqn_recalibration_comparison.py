"""Compare delayed-reward DQN and causal baseline recalibration policies."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.dqn_agent import DQNAgent
from baselines.fixed_schedule import FixedDaysPolicy
from baselines.threshold import ThresholdPolicy
from diagnostics.plots import (
    plot_dqn_recalibration_comparison,
    plot_epsilon_curve,
    plot_training_loss_curve,
    plot_training_reward_curve,
)
from experiments.common import evaluate_policy, make_env, save_csv
from experiments.run_beta_recalibration_timeline import build_beta_schedule, configure_method
from utils.config import load_config
from utils.seed import set_seed


DEFAULT_CONFIG = "configs/beta_recalibration_timeline.yaml"


class DQNPolicy:
    """Greedy evaluation wrapper around one trained agent."""

    name = "dqn"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def reset(self) -> None:
        """DQN has no recurrent policy state."""

    def action(self, state, info: dict) -> int:
        """Select a greedy action."""
        return self.agent.select_action(state, training=False)


def configure_comparison(base_config: dict[str, Any], method: str) -> dict[str, Any]:
    """Build one method-specific causal comparison configuration."""
    config = deepcopy(base_config)
    comparison = config["policy_comparison"]
    config["simulator"] = dict(config["simulator"])
    config["simulator"]["beta_schedule"] = build_beta_schedule(config).tolist()
    config["environment"] = dict(config["environment"])
    config["environment"].update(
        {
            "recalibration_budget": int(comparison["recalibration_budget"]),
            "state_error_mode": "causal_mixed",
            "immediate_error_weight": float(comparison["immediate_error_weight"]),
            "path_error_weight": float(comparison["path_error_weight"]),
        }
    )
    config["dqn"] = dict(config["dqn"])
    config["dqn"]["episodes"] = int(comparison["training_episodes"])
    return configure_method(config, method, fit_window=7)


def make_agent(config: dict[str, Any], seed: int) -> DQNAgent:
    """Create a DQN sized to the configured causal-state environment."""
    env = make_env(config)
    dqn = config["dqn"]
    return DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        hidden_dim=dqn["hidden_dim"],
        learning_rate=dqn["learning_rate"],
        gamma=dqn["gamma"],
        replay_capacity=dqn["replay_capacity"],
        batch_size=dqn["batch_size"],
        epsilon_start=dqn["epsilon_start"],
        epsilon_end=dqn["epsilon_end"],
        epsilon_decay_steps=dqn["epsilon_decay_steps"],
        target_update_interval=dqn["target_update_interval"],
        min_replay_size=dqn["min_replay_size"],
        seed=seed,
    )


def run_delayed_training_episode(
    env,
    agent: DQNAgent,
    seed: int,
) -> dict[str, Any]:
    """Train for one episode, inserting transitions only after path maturity."""
    observation = env.reset(seed=seed)
    pending: dict[int, dict[str, Any]] = {}
    mixed_rewards: list[float] = []
    path_errors: list[float] = []
    immediate_errors: list[float] = []
    losses: list[float] = []
    transitions_added = 0
    terminal_transitions = 0
    final_info: dict[str, Any] = {}
    done = False

    while not done:
        origin_t = int(env.true_simulator.get_state()["t"])
        action = agent.select_action(observation, training=True)
        result = env.step(action)
        info = result.info
        pending[origin_t] = {
            "observation": observation.copy(),
            "action": int(action),
            "next_observation": result.observation.copy(),
            "immediate_error": float(info["forecast_error"]),
        }

        matured_origin = int(info["matured_forecast_origin"])
        if matured_origin >= 0:
            transition = pending.pop(matured_origin)
            mixed_error = float(info["causal_mixed_error"])
            terminal = bool(result.done)
            agent.store_transition(
                transition["observation"],
                transition["action"],
                -mixed_error,
                transition["next_observation"],
                terminal,
            )
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            transitions_added += 1
            terminal_transitions += int(terminal)
            mixed_rewards.append(-mixed_error)
            path_errors.append(float(info["matured_path_error"]))
            immediate_errors.append(float(transition["immediate_error"]))

        observation = result.observation
        final_info = info
        done = result.done

    return {
        "episode_reward": float(np.sum(mixed_rewards)),
        "mean_mixed_error": float(-np.mean(mixed_rewards)),
        "mean_path_error": float(np.mean(path_errors)),
        "mean_immediate_error": float(np.mean(immediate_errors)),
        "mean_loss": float(np.mean(losses)) if losses else np.nan,
        "epsilon": agent.current_epsilon(),
        "budget_used": int(env.initial_budget - env.remaining_budget),
        "episode_length": int(final_info["t"]),
        "transitions_added": transitions_added,
        "dropped_pending_transitions": len(pending),
        "terminal_transitions": terminal_transitions,
    }


def train_delayed_agent(
    config: dict[str, Any],
    seed: int,
    run_dir: Path,
) -> tuple[DQNAgent, pd.DataFrame]:
    """Train and persist one delayed-reward DQN."""
    set_seed(seed)
    env = make_env(config)
    agent = make_agent(config, seed)
    rows = []
    for episode in range(config["dqn"]["episodes"]):
        row = run_delayed_training_episode(env, agent, seed + episode)
        row["episode"] = episode
        rows.append(row)
        if (episode + 1) % 25 == 0:
            print(
                f"method={config['recalibration']['method']} seed={seed} "
                f"episode={episode + 1}/{config['dqn']['episodes']} "
                f"reward={row['episode_reward']:.4f}",
                flush=True,
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


def summarize_comparison_episodes(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Summarize immediate and matured errors for each evaluation episode."""
    group_columns = [
        "recalibration_method_group",
        "policy",
        "training_seed",
        "episode",
        "evaluation_seed",
    ]
    summary = trajectories.groupby(group_columns, as_index=False).agg(
        cumulative_error=("forecast_error", "sum"),
        average_error=("forecast_error", "mean"),
        mean_path_error=("matured_path_error", "mean"),
        mean_mixed_error=("causal_mixed_error", "mean"),
        budget_used=("budget_used", "last"),
        recalibrations=("recalibrated", "sum"),
        dropped_pending_transitions=("dropped_pending_transitions", "last"),
    )
    times = (
        trajectories[trajectories["recalibrated"] == 1]
        .groupby(group_columns)["decision_t"]
        .apply(lambda values: ";".join(str(int(value)) for value in values))
        .reset_index(name="recalibration_times")
    )
    summary = summary.merge(times, on=group_columns, how="left")
    summary["recalibration_times"] = summary["recalibration_times"].fillna("")
    return summary


def aggregate_comparison(episode_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate policy performance across training and evaluation seeds."""
    return (
        episode_summary.groupby(["recalibration_method_group", "policy"], as_index=False)
        .agg(
            cumulative_error_mean=("cumulative_error", "mean"),
            cumulative_error_std=("cumulative_error", "std"),
            average_error_mean=("average_error", "mean"),
            mean_path_error=("mean_path_error", "mean"),
            mean_mixed_error=("mean_mixed_error", "mean"),
            budget_used_mean=("budget_used", "mean"),
            recalibrations_mean=("recalibrations", "mean"),
            evaluation_rows=("evaluation_seed", "size"),
            training_seeds=("training_seed", "nunique"),
        )
        .sort_values(["recalibration_method_group", "policy"])
    )


def paired_differences(episode_summary: pd.DataFrame) -> pd.DataFrame:
    """Pair each DQN episode with both baselines on method and epidemic seed."""
    dqn = episode_summary[episode_summary["policy"] == "dqn"]
    metrics = ["cumulative_error", "average_error", "mean_path_error", "mean_mixed_error", "budget_used"]
    frames = []
    for baseline_name in ["fixed_schedule", "error_threshold"]:
        baseline = episode_summary[episode_summary["policy"] == baseline_name]
        baseline = baseline.drop(columns=["training_seed", "episode", "recalibration_times"])
        merged = dqn.merge(
            baseline,
            on=["recalibration_method_group", "evaluation_seed"],
            suffixes=("_dqn", "_baseline"),
        )
        result = merged[
            ["recalibration_method_group", "training_seed", "evaluation_seed"]
        ].copy()
        result["baseline"] = baseline_name
        for metric in metrics:
            result[f"{metric}_difference"] = merged[f"{metric}_dqn"] - merged[f"{metric}_baseline"]
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def _evaluate(
    config: dict[str, Any],
    policy,
    episodes: int,
    evaluation_seed: int,
    method: str,
    training_seed: int,
) -> pd.DataFrame:
    trajectories = evaluate_policy(make_env(config), policy, episodes, evaluation_seed)
    trajectories["recalibration_method_group"] = method
    trajectories["training_seed"] = int(training_seed)
    trajectories["evaluation_seed"] = trajectories["episode"] + int(evaluation_seed)
    return trajectories


def run(
    config_path: str = DEFAULT_CONFIG,
    methods: list[str] | None = None,
    training_seeds: list[int] | None = None,
    training_episodes: int | None = None,
    evaluation_episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Train DQNs and evaluate paired baselines under both estimators."""
    base_config = load_config(config_path)
    comparison = base_config["policy_comparison"]
    methods = list(comparison["recalibration_methods"] if methods is None else methods)
    training_seeds = list(comparison["training_seeds"] if training_seeds is None else training_seeds)
    training_episodes = int(comparison["training_episodes"] if training_episodes is None else training_episodes)
    evaluation_episodes = int(
        comparison["evaluation_episodes"] if evaluation_episodes is None else evaluation_episodes
    )
    evaluation_seed = int(comparison["evaluation_seed"])
    root = Path(comparison["output_dir"] if output_dir is None else output_dir)
    trajectory_frames = []
    training_frames = []

    for method in methods:
        config = configure_comparison(base_config, method)
        config["dqn"]["episodes"] = training_episodes

        fixed = FixedDaysPolicy(comparison["fixed_days"])
        threshold = ThresholdPolicy(
            error_threshold=float(comparison["error_threshold"]),
            error_history_length=config["environment"]["error_history_length"],
            cooldown=int(comparison["threshold_cooldown"]),
        )
        threshold.name = "error_threshold"
        trajectory_frames.append(
            _evaluate(config, fixed, evaluation_episodes, evaluation_seed, method, -1)
        )
        trajectory_frames.append(
            _evaluate(config, threshold, evaluation_episodes, evaluation_seed, method, -1)
        )

        for seed in training_seeds:
            run_dir = root / method / f"seed_{seed}"
            print(f"training method={method}, seed={seed}", flush=True)
            agent, training_log = train_delayed_agent(config, int(seed), run_dir)
            training_log["recalibration_method_group"] = method
            training_log["training_seed"] = int(seed)
            training_frames.append(training_log)
            trajectory_frames.append(
                _evaluate(
                    config,
                    DQNPolicy(agent),
                    evaluation_episodes,
                    evaluation_seed,
                    method,
                    int(seed),
                )
            )

    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    training_logs = pd.concat(training_frames, ignore_index=True)
    episode_summary = summarize_comparison_episodes(trajectories)
    aggregate = aggregate_comparison(episode_summary)
    paired = paired_differences(episode_summary)
    outputs = {
        "training_logs": training_logs,
        "trajectories": trajectories,
        "episode_summary": episode_summary,
        "aggregate_summary": aggregate,
        "paired_differences": paired,
    }
    for name, frame in outputs.items():
        save_csv(frame, root / f"dqn_recalibration_{name}.csv")
    plot_dqn_recalibration_comparison(episode_summary, trajectories, root / "figures")
    print(aggregate.to_string(index=False))
    print(f"saved comparison results to: {root}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--training-seeds", nargs="+", type=int)
    parser.add_argument("--training-episodes", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(
        config_path=args.config,
        methods=args.methods,
        training_seeds=args.training_seeds,
        training_episodes=args.training_episodes,
        evaluation_episodes=args.evaluation_episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

"""Train and test the two course-project recalibration experiment tiers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.dqn_agent import DQNAgent
from baselines.fixed_schedule import FixedDaysPolicy
from baselines.threshold import ThresholdPolicy
from diagnostics.plots import make_all_plots
from experiments.common import make_agent, make_env, save_csv
from experiments.trajectories import BetaTrajectory, fixed_trajectory, manifest_rows, random_trajectory
from utils.config import load_config
from utils.seed import set_seed

DEFAULT_CONFIG = "configs/course_project.yaml"


class DQNPolicy:
    name = "dqn"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def reset(self) -> None:
        pass

    def action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        return self.agent.select_action(observation, training=False)


def trajectory_for(tier: str, trajectory_seed: int | None) -> BetaTrajectory:
    if tier == "fixed":
        return fixed_trajectory()
    if trajectory_seed is None:
        raise ValueError("random tier requires a trajectory seed")
    return random_trajectory(trajectory_seed)


def training_trajectory_seed(training_seed: int, episode: int) -> int:
    """Place training trajectories in a namespace disjoint from held-out tests."""
    return 10_000_000_000 + int(training_seed) * 1_000_000 + int(episode)


def fixed_policy(tier: str) -> FixedDaysPolicy:
    return FixedDaysPolicy([40, 60, 80, 100, 120, 140] if tier == "fixed" else [30, 60, 90, 120, 150, 180])


def _store_ready(agent: DQNAgent, transition: dict[str, Any], done: bool) -> float | None:
    agent.store_transition(
        transition["observation"], transition["action"], transition["reward"],
        transition["next_observation"], done,
    )
    return agent.update()


def train_episode(env, agent: DQNAgent, epidemic_seed: int) -> dict[str, Any]:
    """Train with one-transition look-behind so only the final mature item is terminal."""
    observation = env.reset(seed=epidemic_seed)
    pending: dict[int, dict[str, Any]] = {}
    ready: dict[str, Any] | None = None
    losses: list[float] = []
    immediate_errors: list[float] = []
    path_errors: list[float] = []
    mixed_errors: list[float] = []
    transitions = 0
    done = False
    final_info: dict[str, Any] = {}
    while not done:
        origin_t = int(env.true_simulator.get_state()["t"])
        action = agent.select_action(observation, training=True)
        result = env.step(action)
        pending[origin_t] = {
            "observation": observation.copy(), "action": int(action),
            "next_observation": result.observation.copy(),
            "immediate_error": float(result.info["forecast_error"]),
        }
        matured_origin = int(result.info["matured_forecast_origin"])
        if matured_origin >= 0:
            source = pending.pop(matured_origin)
            if ready is not None:
                loss = _store_ready(agent, ready, done=False)
                if loss is not None:
                    losses.append(loss)
                transitions += 1
            mixed_error = float(result.info["causal_mixed_error"])
            ready = {
                "observation": source["observation"], "action": source["action"],
                "next_observation": source["next_observation"], "reward": -mixed_error,
            }
            immediate_errors.append(float(source["immediate_error"]))
            path_errors.append(float(result.info["matured_path_error"]))
            mixed_errors.append(mixed_error)
        observation = result.observation
        final_info = result.info
        done = result.done

    if ready is not None:
        loss = _store_ready(agent, ready, done=True)
        if loss is not None:
            losses.append(loss)
        transitions += 1
    return {
        "mean_immediate_error": float(np.mean(immediate_errors)) if immediate_errors else np.nan,
        "mean_path_error": float(np.mean(path_errors)) if path_errors else np.nan,
        "mean_mixed_error": float(np.mean(mixed_errors)) if mixed_errors else np.nan,
        "episode_reward": -float(np.sum(mixed_errors)),
        "mean_loss": float(np.mean(losses)) if losses else np.nan,
        "epsilon": agent.current_epsilon(),
        "budget_used": int(final_info["budget_used"]),
        "episode_length": int(final_info["t"]),
        "termination_reason": final_info["termination_reason"],
        "transitions_added": transitions,
        "dropped_pending_transitions": len(pending),
        "terminal_transitions": int(ready is not None),
    }


def train_agent(
    config: dict,
    tier: str,
    training_seed: int,
    episodes: int,
    run_dir: Path,
) -> tuple[DQNAgent, pd.DataFrame, pd.DataFrame]:
    set_seed(training_seed)
    first_trajectory = trajectory_for(
        tier, None if tier == "fixed" else training_trajectory_seed(training_seed, 0)
    )
    first_env = make_env(config, tier, first_trajectory)
    agent = make_agent(config, first_env.observation_size, training_seed)
    logs: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for episode in range(episodes):
        epidemic_seed = training_seed * 10_000 + episode
        trajectory_seed = None if tier == "fixed" else training_trajectory_seed(training_seed, episode)
        trajectory = trajectory_for(tier, trajectory_seed)
        env = make_env(config, tier, trajectory)
        row = train_episode(env, agent, epidemic_seed)
        row.update(tier=tier, training_seed=training_seed, episode=episode, epidemic_seed=epidemic_seed, trajectory_seed=trajectory_seed)
        logs.append(row)
        manifests.extend(manifest_rows(trajectory, "train", episode, epidemic_seed))
        if (episode + 1) % 25 == 0:
            print(f"tier={tier} seed={training_seed} episode={episode + 1}/{episodes} mixed_error={row['mean_mixed_error']:.6g}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.save(run_dir / "dqn_model.pt")
    log_frame = pd.DataFrame(logs)
    manifest_frame = pd.DataFrame(manifests)
    save_csv(log_frame, run_dir / "training_log.csv")
    save_csv(manifest_frame, run_dir / "training_trajectory_manifest.csv")
    return agent, log_frame, manifest_frame


def evaluate_policy(
    config: dict,
    tier: str,
    policy,
    episodes: int,
    epidemic_seed_base: int,
    trajectory_seed_base: int,
    training_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steps: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for episode in range(episodes):
        epidemic_seed = epidemic_seed_base + episode
        trajectory_seed = None if tier == "fixed" else trajectory_seed_base + episode
        trajectory = trajectory_for(tier, trajectory_seed)
        env = make_env(config, tier, trajectory)
        observation = env.reset(seed=epidemic_seed)
        policy.reset()
        policy_info = {"t": 0, "remaining_budget": env.remaining_budget, "days_since_recalibration": 0, "rolling_mean_forecast_error": 0.0, "rolling_error_trend": 0.0}
        episode_forecasts: list[tuple[int, list[dict[str, float]]]] = []
        done = False
        while not done:
            action = policy.action(observation, policy_info)
            result = env.step(action)
            info = result.info
            one_step = info["surrogate_forecast"][0]
            truth = info["true_state"]
            steps.append({
                "tier": tier, "policy": policy.name, "training_seed": training_seed,
                "episode": episode, "epidemic_seed": epidemic_seed,
                "trajectory_seed": trajectory_seed, "decision_t": info["decision_t"], "t": info["t"],
                "action": info["action"], "recalibrated": int(info["recalibrated"]),
                "remaining_budget": info["remaining_budget"], "budget_used": info["budget_used"],
                "forecast_error": info["forecast_error"], "matured_path_error": info["matured_path_error"],
                "causal_mixed_error": info["causal_mixed_error"], "true_beta": info["true_beta"],
                "beta_hat": info["beta_hat"], "active_infections": info["active_infections"],
                "termination_reason": info["termination_reason"],
                **{f"true_{name}": truth[name] for name in "SEIR"},
                **{f"forecast_{name}": one_step[name] for name in "SEIR"},
            })
            episode_forecasts.append((int(info["decision_t"]), info["surrogate_forecast"]))
            observation = result.observation
            policy_info = info
            done = result.done
        final_t = int(env.history[-1]["t"])
        for origin_t, forecast_path in episode_forecasts:
            if origin_t + env.forecast_horizon > final_t:
                continue
            for offset, forecast in enumerate(forecast_path, start=1):
                target_t = origin_t + offset
                truth = env.history[target_t]
                paths.append({
                    "tier": tier, "policy": policy.name, "training_seed": training_seed,
                    "episode": episode, "epidemic_seed": epidemic_seed,
                    "trajectory_seed": trajectory_seed, "origin_t": origin_t,
                    "offset": offset, "target_t": target_t,
                    **{f"forecast_{name}": forecast[name] for name in "SEIR"},
                    **{f"true_{name}": truth[name] for name in "SEIR"},
                })
        if policy.name == "fixed_schedule":
            manifests.extend(manifest_rows(trajectory, "test", episode, epidemic_seed))
    return pd.DataFrame(steps), pd.DataFrame(paths), pd.DataFrame(manifests)


def summarize_episodes(steps: pd.DataFrame) -> pd.DataFrame:
    keys = ["tier", "policy", "training_seed", "episode", "epidemic_seed", "trajectory_seed"]
    summary = steps.groupby(keys, as_index=False, dropna=False).agg(
        cumulative_one_step_error=("forecast_error", "sum"),
        average_one_step_error=("forecast_error", "mean"),
        mean_path_error=("matured_path_error", "mean"),
        mean_mixed_error=("causal_mixed_error", "mean"),
        episode_length=("t", "max"), budget_used=("budget_used", "last"),
        recalibrations=("recalibrated", "sum"), termination_reason=("termination_reason", "last"),
    )
    times = steps[steps["recalibrated"] == 1].groupby(keys, dropna=False)["decision_t"].apply(lambda x: ";".join(str(int(v)) for v in x)).reset_index(name="recalibration_times")
    summary = summary.merge(times, on=keys, how="left")
    summary["recalibration_times"] = summary["recalibration_times"].fillna("")
    return summary


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["cumulative_one_step_error", "average_one_step_error", "mean_path_error", "mean_mixed_error", "episode_length", "budget_used", "recalibrations"]
    # Average DQN training seeds within each paired test episode before policy-level
    # aggregation, so DQN's three checkpoints do not receive triple the weight.
    balanced = summary.groupby(
        ["tier", "policy", "episode", "epidemic_seed", "trajectory_seed"],
        as_index=False,
        dropna=False,
    )[metrics].mean()
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return balanced.groupby(["tier", "policy"], as_index=False).agg(**aggregations)


def paired_differences(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["cumulative_one_step_error", "average_one_step_error", "mean_path_error", "mean_mixed_error", "budget_used"]
    dqn = summary[summary["policy"] == "dqn"]
    frames = []
    for baseline_name in ["fixed_schedule", "threshold_0.003", "threshold_0.005"]:
        baseline = summary[summary["policy"] == baseline_name].drop(columns=["training_seed", "recalibration_times"])
        merged = dqn.merge(baseline, on=["tier", "episode", "epidemic_seed", "trajectory_seed"], suffixes=("_dqn", "_baseline"))
        result = merged[["tier", "training_seed", "episode", "epidemic_seed", "trajectory_seed"]].copy()
        result["baseline"] = baseline_name
        for metric in metrics:
            result[f"{metric}_difference"] = merged[f"{metric}_dqn"] - merged[f"{metric}_baseline"]
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def run_tier(config: dict, tier: str, output_root: Path, training_episodes: int, evaluation_episodes: int, training_seeds: list[int]) -> None:
    tier_root = output_root / tier
    agents: dict[int, DQNAgent] = {}
    training_logs = []
    training_manifests = []
    for seed in training_seeds:
        agent, log, manifest = train_agent(config, tier, seed, training_episodes, tier_root / "models" / f"seed_{seed}")
        agents[seed] = agent
        training_logs.append(log)
        training_manifests.append(manifest)

    evaluation = config["evaluation"]
    epidemic_base = int(evaluation["epidemic_seed_base"])
    trajectory_base = int(evaluation["trajectory_seed_base"])
    step_frames = []
    path_frames = []
    test_manifests = []
    policies = [fixed_policy(tier)]
    policies.extend(
        ThresholdPolicy(
            threshold,
            error_history_length=config["environment"]["error_history_length"],
            cooldown=evaluation["threshold_cooldown"],
        )
        for threshold in evaluation["thresholds"]
    )
    for policy in policies:
        steps, paths, manifests = evaluate_policy(config, tier, policy, evaluation_episodes, epidemic_base, trajectory_base, -1)
        step_frames.append(steps)
        path_frames.append(paths)
        if not manifests.empty:
            test_manifests.append(manifests)
    for seed, agent in agents.items():
        steps, paths, _ = evaluate_policy(config, tier, DQNPolicy(agent), evaluation_episodes, epidemic_base, trajectory_base, seed)
        step_frames.append(steps)
        path_frames.append(paths)

    steps = pd.concat(step_frames, ignore_index=True)
    paths = pd.concat(path_frames, ignore_index=True)
    summary = summarize_episodes(steps)
    aggregate = aggregate_summary(summary)
    paired = paired_differences(summary)
    save_csv(pd.concat(training_logs, ignore_index=True), tier_root / "training_logs.csv")
    save_csv(pd.concat(training_manifests, ignore_index=True), tier_root / "training_trajectory_manifest.csv")
    if test_manifests:
        save_csv(pd.concat(test_manifests, ignore_index=True), tier_root / "test_trajectory_manifest.csv")
    save_csv(steps, tier_root / "test_step_trajectories.csv")
    save_csv(paths, tier_root / "test_forecast_paths.csv")
    save_csv(summary, tier_root / "test_episode_summary.csv")
    save_csv(aggregate, tier_root / "test_aggregate_summary.csv")
    save_csv(paired, tier_root / "test_paired_differences.csv")
    make_all_plots(summary, steps, paths, tier_root / "figures")
    print(aggregate.to_string(index=False))


def run(config_path: str, tiers: list[str], output_dir: str, training_episodes: int | None = None, evaluation_episodes: int | None = None, training_seeds: list[int] | None = None) -> None:
    config = load_config(config_path)
    training_episodes = int(training_episodes or config["dqn"]["training_episodes"])
    evaluation_episodes = int(evaluation_episodes or config["evaluation"]["episodes"])
    training_seeds = list(training_seeds or config["dqn"]["training_seeds"])
    for tier in tiers:
        if tier not in {"fixed", "random"}:
            raise ValueError("tiers must contain fixed and/or random")
        run_tier(config, tier, Path(output_dir), training_episodes, evaluation_episodes, training_seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tiers", nargs="+", choices=["fixed", "random"], default=["fixed", "random"])
    parser.add_argument("--output-dir", default="results/course_project")
    parser.add_argument("--training-episodes", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--training-seeds", nargs="+", type=int)
    args = parser.parse_args()
    run(args.config, args.tiers, args.output_dir, args.training_episodes, args.evaluation_episodes, args.training_seeds)


if __name__ == "__main__":
    main()

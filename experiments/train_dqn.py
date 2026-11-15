"""Train a DQN recalibration policy from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agents.dqn_agent import DQNAgent
from diagnostics.plots import plot_epsilon_curve, plot_training_loss_curve, plot_training_reward_curve
from experiments.common import make_env, save_csv
from utils.config import ensure_output_dirs, load_config
from utils.seed import set_seed


def train(config_path: str) -> None:
    """Run DQN training and save logs/checkpoint."""
    config = load_config(config_path)
    ensure_output_dirs(config)
    set_seed(config["seed"])

    env = make_env(config)
    dqn_cfg = config["dqn"]
    agent = DQNAgent(
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
        seed=config["seed"],
    )

    rows = []
    global_step = 0
    for episode in range(dqn_cfg["episodes"]):
        observation = env.reset(seed=config["seed"] + episode)
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
            global_step += 1

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
                "budget_utilization": (env.budget - env.remaining_budget) / max(env.budget, 1),
                "num_recalibrations": env.budget - env.remaining_budget,
                "recalibrations_used": env.budget - env.remaining_budget,
                "episode_length": int(final_info.get("t", 0)),
                "termination_reason": final_info.get("termination_reason", ""),
                "final_active_infections": final_info.get("active_infections", np.nan),
            }
        )
        if (episode + 1) % 25 == 0:
            print(f"episode {episode + 1}/{dqn_cfg['episodes']} reward={episode_reward:.4f}")

    results_dir = Path(config["outputs"]["results_dir"])
    figures_dir = Path(config["outputs"].get("figures_dir", results_dir / "figures"))
    model_path = results_dir / "dqn_model.pt"
    log_path = results_dir / "dqn_training_log.csv"

    agent.save(model_path)
    training_log = pd.DataFrame(rows)
    save_csv(training_log, log_path)
    plot_training_reward_curve(training_log, figures_dir / "dqn_training_reward.png")
    plot_training_loss_curve(training_log, figures_dir / "dqn_training_loss.png")
    plot_epsilon_curve(training_log, figures_dir / "dqn_epsilon.png")
    print(f"saved model: {model_path}")
    print(f"saved training log: {log_path}")
    print(f"saved training plots: {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

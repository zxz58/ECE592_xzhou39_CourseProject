"""Regression tests for the focused course-project experiment pipeline."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from baselines.fixed_schedule import FixedDaysPolicy
from baselines.threshold import ThresholdPolicy
from experiments.common import make_agent, make_env
from experiments.run_course_experiments import run, train_episode, training_trajectory_seed
from experiments.trajectories import fixed_trajectory, random_trajectory
from utils.config import load_config


class TrajectoryTests(unittest.TestCase):
    def test_fixed_trajectory(self) -> None:
        trajectory = fixed_trajectory()
        self.assertEqual(len(trajectory.schedule), 160)
        self.assertEqual(len(trajectory.events), 4)
        self.assertTrue(np.allclose(trajectory.schedule[0:20], 0.30))
        self.assertAlmostEqual(trajectory.schedule[49], 0.60)
        self.assertAlmostEqual(trajectory.schedule[65], 0.25)
        self.assertAlmostEqual(trajectory.schedule[85], 0.70)
        self.assertAlmostEqual(trajectory.schedule[134], 0.35)
        self.assertTrue(np.allclose(trajectory.schedule[135:], 0.35))

    def test_random_trajectory_constraints_and_reproducibility(self) -> None:
        first = random_trajectory(9876)
        second = random_trajectory(9876)
        self.assertTrue(np.array_equal(first.schedule, second.schedule))
        self.assertEqual(first.events, second.events)
        self.assertEqual(len(first.events), 4)
        previous_end = -1
        previous_beta = 0.30
        for event in first.events:
            gap = event.start - previous_end - 1
            self.assertGreaterEqual(gap, 2)
            self.assertLessEqual(gap, 30)
            if event.duration_group == "short":
                self.assertTrue(1 <= event.duration <= 10)
            else:
                self.assertTrue(11 <= event.duration <= 40)
            self.assertTrue(0.15 <= event.end_beta <= 0.85)
            self.assertGreaterEqual(abs(event.end_beta - previous_beta), 0.10)
            previous_end = event.end
            previous_beta = event.end_beta
        self.assertLess(first.events[-1].end, 240)

    def test_training_and_testing_trajectory_seeds_are_disjoint(self) -> None:
        training = {training_trajectory_seed(seed, episode) for seed in (123, 456, 789) for episode in range(500)}
        testing = {1_000_000 + episode for episode in range(100)}
        self.assertTrue(training.isdisjoint(testing))


class EnvironmentAndPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/course_project.yaml")

    def test_oracle_budget_and_fixed_termination(self) -> None:
        env = make_env(self.config, "fixed", fixed_trajectory())
        env.reset(seed=42)
        for _ in range(10):
            result = env.step(1)
        self.assertEqual(result.info["budget_used"], 6)
        self.assertEqual(result.info["beta_hat"], result.info["true_beta"])
        while not result.done:
            result = env.step(0)
        self.assertEqual(result.info["t"], 160)
        self.assertEqual(result.info["termination_reason"], "fixed_horizon")
        self.assertEqual(sum(env.true_simulator.get_state()[name] for name in "SEIR"), 100000)

        change_env = make_env(self.config, "fixed", fixed_trajectory())
        change_env.reset(seed=42)
        for _ in range(65):
            change_env.step(0)
        changed = change_env.step(1)
        self.assertAlmostEqual(changed.info["true_beta"], 0.25)
        self.assertAlmostEqual(changed.info["beta_hat"], 0.25)

    def test_random_safety_cap_without_extinction(self) -> None:
        config = deepcopy(self.config)
        config["environment"]["extinction_threshold"] = -1
        env = make_env(config, "random", random_trajectory(1234))
        env.reset(seed=5)
        result = None
        while result is None or not result.done:
            result = env.step(0)
        self.assertEqual(result.info["t"], 240)
        self.assertEqual(result.info["termination_reason"], "max_steps")

    def test_baseline_trigger_rules(self) -> None:
        fixed = FixedDaysPolicy([30, 60, 90, 120, 150, 180])
        self.assertEqual(fixed.action(None, {"t": 30, "remaining_budget": 6}), 1)
        self.assertEqual(fixed.action(None, {"t": 31, "remaining_budget": 6}), 0)
        threshold = ThresholdPolicy(0.003, cooldown=7)
        self.assertEqual(threshold.action(None, {"remaining_budget": 6, "days_since_recalibration": 6, "rolling_mean_forecast_error": 0.01}), 0)
        self.assertEqual(threshold.action(None, {"remaining_budget": 6, "days_since_recalibration": 7, "rolling_mean_forecast_error": 0.003}), 1)
        self.assertEqual(threshold.action(None, {"remaining_budget": 0, "days_since_recalibration": 20, "rolling_mean_forecast_error": 0.01}), 0)

    def test_delayed_replay_has_one_terminal_transition(self) -> None:
        env = make_env(self.config, "fixed", fixed_trajectory())
        agent = make_agent(self.config, env.observation_size, 7)
        row = train_episode(env, agent, epidemic_seed=99)
        self.assertEqual(row["transitions_added"], 160 - self.config["environment"]["forecast_horizon"] + 1)
        self.assertEqual(int(agent.replay_buffer.dones[: len(agent.replay_buffer)].sum()), 1)
        self.assertEqual(row["dropped_pending_transitions"], 6)


class SmokeTest(unittest.TestCase):
    def test_small_end_to_end_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run(
                config_path="configs/course_project.yaml",
                tiers=["fixed", "random"],
                output_dir=directory,
                training_episodes=1,
                evaluation_episodes=1,
                training_seeds=[123],
            )
            for tier in ("fixed", "random"):
                root = Path(directory) / tier
                self.assertTrue((root / "test_step_trajectories.csv").exists())
                self.assertTrue((root / "test_forecast_paths.csv").exists())
                self.assertTrue((root / "test_aggregate_summary.csv").exists())
                self.assertTrue((root / "figures" / "prediction_performance.png").exists())


if __name__ == "__main__":
    unittest.main()

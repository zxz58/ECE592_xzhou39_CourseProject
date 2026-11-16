"""Focused figures for the two course-project experiment tiers."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache")))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICIES = ["fixed_schedule", "threshold_0.003", "threshold_0.005", "dqn"]
COLORS = {"fixed_schedule": "#4c78a8", "threshold_0.003": "#f58518", "threshold_0.005": "#b279a2", "dqn": "#54a24b"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _change_spans(data: pd.DataFrame) -> list[tuple[int, int]]:
    """Infer contiguous beta-change regions for representative diagnostics."""
    beta = data.sort_values("decision_t").drop_duplicates("decision_t").set_index("decision_t")["true_beta"]
    changed = beta.diff().abs().fillna(0.0) > 1e-12
    days = beta.index[changed].to_list()
    if not days:
        return []
    groups: list[list[int]] = [[int(days[0])]]
    for day in days[1:]:
        if int(day) == groups[-1][-1] + 1:
            groups[-1].append(int(day))
        else:
            groups.append([int(day)])
    return [(max(group[0] - 1, 0), group[-1]) for group in groups]


def plot_prediction_performance(summary: pd.DataFrame, figures_dir: Path) -> None:
    balanced = summary.groupby(["policy", "episode"], as_index=False).mean(numeric_only=True)
    specs = [
        ("average_one_step_error", "Average daily one-step SEIR MAE"),
        ("mean_path_error", "Mean complete 7-day path SEIR MAE"),
        ("mean_mixed_error", "Mean mixed error (0.3 one-step + 0.7 path)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (column, label) in zip(axes, specs):
        values = [balanced.loc[balanced["policy"] == policy, column].dropna().to_numpy() for policy in POLICIES]
        boxes = ax.boxplot(values, tick_labels=POLICIES, patch_artist=True, showfliers=False)
        for box, policy in zip(boxes["boxes"], POLICIES):
            box.set_facecolor(COLORS[policy])
            box.set_alpha(0.78)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    _save(fig, figures_dir / "prediction_performance.png")


def plot_budget_and_timing(summary: pd.DataFrame, steps: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    means = summary.groupby("policy")["budget_used"].mean().reindex(POLICIES)
    stds = summary.groupby("policy")["budget_used"].std().reindex(POLICIES)
    axes[0].bar(POLICIES, means, yerr=stds, color=[COLORS[p] for p in POLICIES], capsize=4)
    axes[0].set_ylabel("Budget used (maximum 6)")
    axes[0].tick_params(axis="x", rotation=25)
    bins = np.arange(0, int(steps["t"].max()) + 11, 10)
    for policy in POLICIES:
        times = steps.loc[(steps["policy"] == policy) & (steps["recalibrated"] == 1), "decision_t"]
        episode_count = steps.loc[steps["policy"] == policy, ["training_seed", "episode"]].drop_duplicates().shape[0]
        weights = np.full(len(times), 1.0 / max(episode_count, 1))
        axes[1].hist(times, bins=bins, weights=weights, histtype="step", linewidth=2, color=COLORS[policy], label=policy)
    axes[1].set_xlabel("Recalibration decision day")
    axes[1].set_ylabel("Mean recalibrations per episode")
    axes[1].legend(fontsize=8)
    _save(fig, figures_dir / "budget_and_timing.png")


def plot_beta_tracking(steps: pd.DataFrame, figures_dir: Path, episode: int = 0) -> None:
    data = steps[steps["episode"] == episode]
    fig, axes = plt.subplots(len(POLICIES), 1, figsize=(11, 9), sharex=True, sharey=True)
    for ax, policy in zip(axes, POLICIES):
        policy_data = data[data["policy"] == policy]
        policy_data = policy_data[policy_data["training_seed"] == policy_data["training_seed"].min()]
        ax.plot(policy_data["decision_t"], policy_data["true_beta"], color="black", label="true beta")
        ax.plot(policy_data["decision_t"], policy_data["beta_hat"], color=COLORS[policy], linestyle="--", label="surrogate beta")
        for start, end in _change_spans(policy_data):
            ax.axvspan(start, end, color="#f59e0b", alpha=0.12)
        for day in policy_data.loc[policy_data["recalibrated"] == 1, "decision_t"]:
            ax.axvline(day, color="#c2410c", alpha=0.22)
        ax.set_ylabel(policy)
        ax.grid(alpha=0.18)
    axes[0].legend(ncols=2, fontsize=8)
    axes[-1].set_xlabel("Day")
    _save(fig, figures_dir / "beta_tracking.png")


def plot_representative_seir(paths: pd.DataFrame, figures_dir: Path, episode: int = 0) -> None:
    data = paths[paths["episode"] == episode]
    # One origin every seven days keeps the aligned path plot readable.
    data = data[data["origin_t"] % 7 == 0]
    fig, axes = plt.subplots(len(POLICIES), 4, figsize=(17, 10), sharex=True)
    for row, policy in enumerate(POLICIES):
        policy_data = data[data["policy"] == policy]
        policy_data = policy_data[policy_data["training_seed"] == policy_data["training_seed"].min()]
        for column, compartment in enumerate("SEIR"):
            ax = axes[row, column]
            truth = policy_data.groupby("target_t")[f"true_{compartment}"].first()
            ax.plot(truth.index, truth.values, color="black", linewidth=1.5, label="truth")
            for _, group in policy_data.groupby("origin_t"):
                ax.plot(group["target_t"], group[f"forecast_{compartment}"], color=COLORS[policy], alpha=0.22)
            if row == 0:
                ax.set_title(compartment)
            if column == 0:
                ax.set_ylabel(policy)
            ax.grid(alpha=0.14)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Target day")
    _save(fig, figures_dir / "representative_seir_forecasts.png")


def make_all_plots(summary: pd.DataFrame, steps: pd.DataFrame, paths: pd.DataFrame, figures_dir: Path) -> None:
    plot_prediction_performance(summary, figures_dir)
    plot_budget_and_timing(summary, steps, figures_dir)
    plot_beta_tracking(steps, figures_dir)
    plot_representative_seir(paths, figures_dir)

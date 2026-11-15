"""Diagnostic plotting helpers."""

from __future__ import annotations

import os
from pathlib import Path

# Keep matplotlib's font cache out of experiment output directories.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

COMPARISON_BAR_COLOR = "#4c78a8"


def _save(
    fig: plt.Figure,
    path: str | Path,
    layout_kwargs: dict | None = None,
) -> None:
    """Save a figure and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**(layout_kwargs or {}))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _episode_data(trajectories: pd.DataFrame, episode: int = 0, policy: str | None = None) -> pd.DataFrame:
    """Select one episode, optionally for one policy."""
    data = trajectories[trajectories["episode"] == episode]
    if policy is not None:
        data = data[data["policy"] == policy]
    if data.empty:
        return trajectories.head(0)
    return data.sort_values("t")


def _mark_recalibrations(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw vertical markers at recalibration times."""
    recalibration_times = data.loc[data["recalibrated"] == 1, "t"].drop_duplicates()
    for time in recalibration_times:
        ax.axvline(time, color="#c2410c", alpha=0.18, linewidth=1)


def _mark_surge_period(
    ax: plt.Axes,
    surge_start: int | None,
    surge_end: int | None,
    label: bool = False,
) -> None:
    """Draw the beta-surge window on plots with day on the x-axis."""
    if surge_start is None or surge_end is None:
        return
    span_label = "surge period" if label else None
    ax.axvspan(surge_start, surge_end, color="#f59e0b", alpha=0.12, label=span_label)
    ax.axvline(surge_start, color="#92400e", linestyle="--", linewidth=1, alpha=0.75)
    ax.axvline(surge_end, color="#92400e", linestyle="--", linewidth=1, alpha=0.75)


def plot_true_seir_curves(
    trajectories: pd.DataFrame,
    path: str | Path,
    episode: int = 0,
    policy: str | None = None,
) -> None:
    """Save true S/E/I/R curves with recalibration markers."""
    data = _episode_data(trajectories, episode=episode, policy=policy)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["t"], data["true_S"], label="S")
    ax.plot(data["t"], data["true_E"], label="E")
    ax.plot(data["t"], data["true_I"], label="I")
    ax.plot(data["t"], data["true_R"], label="R")
    _mark_recalibrations(ax, data)
    ax.set_title("True SEIR curves")
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.legend(ncols=4)
    _save(fig, path)


def plot_beta_tracking(
    trajectories: pd.DataFrame,
    path: str | Path,
    episode: int = 0,
    policy: str | None = None,
) -> None:
    """Save true beta versus surrogate beta_hat over time."""
    data = _episode_data(trajectories, episode=episode, policy=policy)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["t"], data["true_beta"], color="black", linestyle="--", label="true beta")
    ax.plot(data["t"], data["beta_hat"], color="#2563eb", label="beta_hat")
    _mark_recalibrations(ax, data)
    ax.set_title("True beta vs surrogate beta_hat")
    ax.set_xlabel("Day")
    ax.set_ylabel("Beta")
    ax.legend()
    _save(fig, path)


def plot_forecast_error(
    trajectories: pd.DataFrame,
    path: str | Path,
    episode: int = 0,
    policy: str | None = None,
) -> None:
    """Save forecast error over time with recalibration markers."""
    data = _episode_data(trajectories, episode=episode, policy=policy)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data["t"], data["forecast_error"], color="#7c3aed")
    _mark_recalibrations(ax, data)
    ax.set_title("Forecast error over time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Forecast error")
    _save(fig, path)


def plot_cumulative_error_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    """Save cumulative forecast error comparison across policies."""
    means = summary.groupby("policy")["cumulative_error"].mean().sort_values()
    errors = summary.groupby("policy")["cumulative_error"].std().reindex(means.index)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    means.plot(kind="bar", yerr=errors, ax=ax, color=COMPARISON_BAR_COLOR, capsize=4)
    ax.set_ylabel("Cumulative forecast error")
    ax.set_xlabel("Policy")
    ax.set_title("Cumulative forecast error by policy")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, path)


def plot_budget_usage_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    """Save budget usage comparison across policies."""
    means = summary.groupby("policy")["budget_used"].mean().sort_values()
    errors = summary.groupby("policy")["budget_used"].std().reindex(means.index)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    means.plot(kind="bar", yerr=errors, ax=ax, color=COMPARISON_BAR_COLOR, capsize=4)
    ax.set_ylabel("Budget used")
    ax.set_xlabel("Policy")
    ax.set_title("Budget usage by policy")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, path)


def plot_episode_length_distribution(summary: pd.DataFrame, path: str | Path) -> None:
    """Save episode length distribution by policy."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    summary.boxplot(column="episode_length", by="policy", ax=ax, rot=25)
    ax.set_title("Episode length distribution")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Episode length")
    fig.suptitle("")
    _save(fig, path)


def plot_average_daily_error_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    """Save average daily forecast error comparison."""
    means = summary.groupby("policy")["average_forecast_error"].mean().sort_values()
    errors = summary.groupby("policy")["average_forecast_error"].std().reindex(means.index)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    means.plot(kind="bar", yerr=errors, ax=ax, color=COMPARISON_BAR_COLOR, capsize=4)
    ax.set_ylabel("Average daily forecast error")
    ax.set_xlabel("Policy")
    ax.set_title("Average daily forecast error by policy")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, path)


def plot_normalized_recalibration_timing(
    summary: pd.DataFrame,
    path: str | Path,
    surge_start: int | None = None,
    surge_end: int | None = None,
) -> None:
    """Save recalibration timing normalized by episode progress."""
    rows = []
    for _, row in summary.iterrows():
        if not row.get("recalibration_times"):
            continue
        episode_length = max(float(row["episode_length"]), 1.0)
        for time in str(row["recalibration_times"]).split(";"):
            if time:
                rows.append({"policy": row["policy"], "normalized_time": float(time) / episode_length})
    timing = pd.DataFrame(rows)
    if timing.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    timing.boxplot(column="normalized_time", by="policy", ax=ax, rot=25)
    ax.set_title("Recalibration timing by epidemic progress")
    ax.set_xlabel("Policy")
    ax.set_ylabel("t / episode length")
    if surge_start is not None and surge_end is not None:
        reference_length = max(float(summary["episode_length"].mean()), 1.0)
        start = min(float(surge_start) / reference_length, 1.0)
        end = min(float(surge_end) / reference_length, 1.0)
        ax.axhspan(start, end, color="#f59e0b", alpha=0.12, label="surge period")
        ax.axhline(start, color="#92400e", linestyle="--", linewidth=1, alpha=0.75, label="surge start")
        ax.axhline(end, color="#92400e", linestyle=":", linewidth=1, alpha=0.75, label="surge end")
        ax.legend(loc="upper right")
    fig.suptitle("")
    _save(fig, path)


def plot_training_reward_curve(training_log: pd.DataFrame, path: str | Path) -> None:
    """Save DQN training cumulative reward curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    y_col = "cumulative_reward" if "cumulative_reward" in training_log else "episode_reward"
    ax.plot(training_log["episode"], training_log[y_col], color="#2f855a")
    ax.set_title("DQN training reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    _save(fig, path)


def plot_epsilon_curve(training_log: pd.DataFrame, path: str | Path) -> None:
    """Save DQN epsilon schedule over training."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(training_log["episode"], training_log["epsilon"], color="#b7791f")
    ax.set_title("DQN epsilon schedule")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    _save(fig, path)


def plot_training_loss_curve(training_log: pd.DataFrame, path: str | Path) -> None:
    """Save DQN mean training loss by episode."""
    if "mean_loss" not in training_log:
        return
    data = training_log[["episode", "mean_loss"]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data["episode"], data["mean_loss"], color="#b7791f")
    ax.set_title("DQN training loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean loss")
    _save(fig, path)


def plot_training_curve(training_log: pd.DataFrame, path: str | Path) -> None:
    """Compatibility plot with reward and loss curves."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    y_col = "cumulative_reward" if "cumulative_reward" in training_log else "episode_reward"
    axes[0].plot(training_log["episode"], training_log[y_col], color="#2f855a")
    axes[0].set_ylabel("Cumulative reward")
    axes[1].plot(training_log["episode"], training_log["mean_loss"], color="#b7791f")
    axes[1].set_ylabel("Mean loss")
    axes[1].set_xlabel("Episode")
    _save(fig, path)


def plot_robustness_cumulative_error(summary: pd.DataFrame, path: str | Path) -> None:
    """Save mean +/- std cumulative error over seeds/scenarios."""
    if "scenario" in summary.columns:
        grouped = summary.groupby(["scenario", "policy"], as_index=False)["cumulative_error"].agg(["mean", "std"])
        pivot_mean = grouped.pivot(index="scenario", columns="policy", values="mean")
        pivot_std = grouped.pivot(index="scenario", columns="policy", values="std")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(pivot_mean.index))
        for policy in pivot_mean.columns:
            ax.errorbar(x, pivot_mean[policy], yerr=pivot_std[policy], marker="o", capsize=3, label=policy)
        ax.set_xticks(list(x))
        ax.set_xticklabels(pivot_mean.index, rotation=20)
        ax.set_xlabel("Scenario")
    else:
        grouped = summary.groupby("policy")["cumulative_error"].agg(["mean", "std"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o", capsize=4)
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("Policy")
    ax.set_ylabel("Cumulative forecast error")
    ax.set_title("Robustness: mean +/- std cumulative error")
    ax.legend(loc="best") if "scenario" in summary.columns else None
    _save(fig, path)


def plot_budget_sweep(summary: pd.DataFrame, error_path: str | Path, usage_path: str | Path) -> None:
    """Save budget sweep error and budget usage plots."""
    grouped = summary.groupby(["budget", "policy"], as_index=False).agg(
        cumulative_error_mean=("cumulative_error", "mean"),
        budget_used_mean=("budget_used", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy, data in grouped.groupby("policy"):
        data = data.sort_values("budget")
        ax.plot(data["budget"], data["cumulative_error_mean"], marker="o", label=policy)
    ax.set_title("Cumulative forecast error vs budget")
    ax.set_xlabel("Budget allowed")
    ax.set_ylabel("Cumulative forecast error")
    ax.legend()
    _save(fig, error_path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy, data in grouped.groupby("policy"):
        data = data.sort_values("budget")
        ax.plot(data["budget"], data["budget_used_mean"], marker="o", label=policy)
    ax.plot(sorted(grouped["budget"].unique()), sorted(grouped["budget"].unique()), "--", color="black", label="allowed")
    ax.set_title("Budget used vs budget allowed")
    ax.set_xlabel("Budget allowed")
    ax.set_ylabel("Budget used")
    ax.legend()
    _save(fig, usage_path)


def plot_error_history_sweep(summary: pd.DataFrame, path: str | Path) -> None:
    """Save DQN cumulative error by error-history length."""
    data = summary.sort_values("error_history_length")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        data["error_history_length"],
        data["cumulative_error_mean"],
        yerr=data["cumulative_error_std"],
        marker="o",
        color=COMPARISON_BAR_COLOR,
        capsize=4,
        linewidth=2,
    )
    ax.set_title("DQN cumulative error vs error-history length")
    ax.set_xlabel("Error-history length K")
    ax.set_ylabel("Cumulative forecast error")
    ax.set_xticks(data["error_history_length"].tolist())
    _save(fig, path)


TIMELINE_METHOD_COLORS = {
    "oracle": "#1b1b1b",
    "oracle_noisy": "#4c78a8",
    "moment_match": "#f58518",
    "grid_fit": "#54a24b",
    "moving_average": "#b279a2",
    "partial_update": "#e45756",
}


def _method_color(method: str, index: int) -> str:
    if method in TIMELINE_METHOD_COLORS:
        return TIMELINE_METHOD_COLORS[method]
    return plt.get_cmap("tab10")(index % 10)


def _timeline_reference(step_trajectories: pd.DataFrame) -> pd.DataFrame:
    """Select one copy of the shared true beta timeline."""
    first_method = step_trajectories["method"].iloc[0]
    first_seed = step_trajectories.loc[step_trajectories["method"] == first_method, "seed"].iloc[0]
    return (
        step_trajectories[
            (step_trajectories["method"] == first_method) & (step_trajectories["seed"] == first_seed)
        ][["day", "true_beta"]]
        .sort_values("day")
        .drop_duplicates("day")
    )


def plot_beta_timeline_benchmark(
    step_trajectories: pd.DataFrame,
    trigger_results: pd.DataFrame,
    figures_dir: str | Path,
    method_order: list[str],
    jump_days: dict[str, int],
) -> None:
    """Save the fixed-timeline beta and infectious-case plots."""
    figures_dir = Path(figures_dir)
    _plot_timeline_beta_trajectories(step_trajectories, trigger_results, figures_dir, method_order)
    _plot_timeline_signed_error_boxes(trigger_results, figures_dir, method_order, jump_days)
    _plot_timeline_exposed_trajectories(step_trajectories, figures_dir, method_order)
    _plot_timeline_infectious_7d_forecasts(step_trajectories, figures_dir, method_order)


def _plot_timeline_beta_trajectories(
    step_trajectories: pd.DataFrame,
    trigger_results: pd.DataFrame,
    figures_dir: Path,
    method_order: list[str],
) -> None:
    reference = _timeline_reference(step_trajectories)
    n_methods = len(method_order)
    n_columns = 2
    n_rows = int(np.ceil(n_methods / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(13, 3.4 * n_rows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    y_max = max(float(reference["true_beta"].max()), float(trigger_results["applied_beta"].quantile(0.99)))
    for index, method in enumerate(method_order):
        ax = axes_array[index]
        data = trigger_results[trigger_results["method"] == method]
        summary = data.groupby("trigger_day")["applied_beta"].agg(
            median="median",
            p10=lambda values: values.quantile(0.10),
            p90=lambda values: values.quantile(0.90),
        )
        color = _method_color(method, index)
        ax.plot(reference["day"], reference["true_beta"], color="#202020", linewidth=2, label="true beta")
        ax.errorbar(
            summary.index,
            summary["median"],
            yerr=np.vstack([summary["median"] - summary["p10"], summary["p90"] - summary["median"]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4,
            label="median applied beta (10th-90th percentile)",
        )
        ax.set_title(method.replace("_", " "))
        ax.set_xlim(0, int(reference["day"].max()))
        ax.set_ylim(0, max(0.8, y_max * 1.08))
        ax.grid(alpha=0.18)
        if index % n_columns == 0:
            ax.set_ylabel("Beta")
        if index >= n_methods - n_columns:
            ax.set_xlabel("Day")
    for ax in axes_array[n_methods:]:
        ax.set_visible(False)
    axes_array[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("True and applied beta at fixed recalibration triggers", y=0.995)
    _save(fig, figures_dir / "beta_trajectory_by_method.png")


def _plot_timeline_signed_error_boxes(
    trigger_results: pd.DataFrame,
    figures_dir: Path,
    method_order: list[str],
    jump_days: dict[str, int],
) -> None:
    trigger_days = sorted(trigger_results["trigger_day"].unique())
    offsets = np.linspace(-1.25, 1.25, len(method_order))
    fig, ax = plt.subplots(figsize=(22, 7))
    for index, method in enumerate(method_order):
        method_data = trigger_results[trigger_results["method"] == method]
        values = [
            method_data.loc[method_data["trigger_day"] == day, "signed_error_pct"].dropna().to_numpy()
            for day in trigger_days
        ]
        boxes = ax.boxplot(
            values,
            positions=[day + offsets[index] for day in trigger_days],
            widths=0.38,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        color = _method_color(method, index)
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.72)
        for median in boxes["medians"]:
            median.set_color("white")
            median.set_linewidth(1.2)
    ax.axhline(0, color="#202020", linewidth=1)
    for event, day in jump_days.items():
        ax.axvline(day, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(day + 0.4, ax.get_ylim()[1] * 0.92, event.replace("_", " "), rotation=90, va="top", fontsize=8)
    ax.set_xticks(trigger_days)
    ax.set_xticklabels([str(day) for day in trigger_days])
    ax.set_xlabel("Recalibration trigger day")
    ax.set_ylabel("Signed applied-beta error (%)")
    ax.set_title("Applied-beta error distribution at each fixed trigger")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        handles=[Patch(facecolor=_method_color(method, index), label=method) for index, method in enumerate(method_order)],
        ncols=3,
        loc="upper right",
        fontsize=8,
    )
    _save(fig, figures_dir / "signed_error_boxplot.png")


def _trajectory_quantiles(
    data: pd.DataFrame,
    day_column: str,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Summarize a trajectory across paired repetitions."""
    return data.groupby(day_column)[value_column].agg(
        **{
            f"{prefix}_median": "median",
            f"{prefix}_p10": lambda values: values.quantile(0.10),
            f"{prefix}_p90": lambda values: values.quantile(0.90),
        }
    )


def _plot_compartment_panels(
    summaries: dict[str, pd.DataFrame],
    figures_dir: Path,
    method_order: list[str],
    ylabel: str,
    title: str,
    filename: str,
    true_label: str,
    surrogate_label: str,
) -> None:
    """Plot paired true and surrogate trajectory summaries by method."""
    y_max = max(
        float(summary[["true_p90", "surrogate_p90"]].max().max())
        for summary in summaries.values()
    )
    n_methods = len(method_order)
    n_columns = 2
    n_rows = int(np.ceil(n_methods / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(13, 3.4 * n_rows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    for index, method in enumerate(method_order):
        ax = axes_array[index]
        summary = summaries[method]
        days = summary.index.to_numpy()
        color = _method_color(method, index)
        surrogate_color = "#2563eb" if method == "oracle" else color
        ax.fill_between(days, summary["true_p10"], summary["true_p90"], color="#202020", alpha=0.10)
        ax.fill_between(
            days,
            summary["surrogate_p10"],
            summary["surrogate_p90"],
            color=surrogate_color,
            alpha=0.16,
        )
        ax.plot(days, summary["true_median"], color="#202020", linewidth=1.8, label=true_label)
        ax.plot(
            days,
            summary["surrogate_median"],
            color=surrogate_color,
            linewidth=1.5,
            linestyle="--",
            label=surrogate_label,
        )
        ax.set_title(method.replace("_", " "))
        ax.set_xlim(int(days.min()), int(days.max()))
        ax.set_ylim(0, y_max * 1.08 if y_max > 0 else 1.0)
        ax.grid(alpha=0.18)
        if index % n_columns == 0:
            ax.set_ylabel(ylabel)
        if index >= n_methods - n_columns:
            ax.set_xlabel("Day")
    for ax in axes_array[n_methods:]:
        ax.set_visible(False)
    axes_array[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, y=0.995)
    _save(
        fig,
        figures_dir / filename,
        layout_kwargs={"rect": (0, 0, 1, 0.97), "h_pad": 2.0},
    )


def _plot_timeline_exposed_trajectories(
    step_trajectories: pd.DataFrame,
    figures_dir: Path,
    method_order: list[str],
) -> None:
    """Compare paired true and one-step surrogate exposed counts."""
    summaries: dict[str, pd.DataFrame] = {}
    for method in method_order:
        data = step_trajectories[step_trajectories["method"] == method]
        summaries[method] = _trajectory_quantiles(data, "next_day", "true_E", "true").join(
            _trajectory_quantiles(data, "next_day", "surrogate_E", "surrogate"),
            how="inner",
        )
    _plot_compartment_panels(
        summaries,
        figures_dir,
        method_order,
        ylabel="Exposed cases (E)",
        title="True and surrogate exposed cases",
        filename="exposed_trajectory_by_method.png",
        true_label="true E",
        surrogate_label="surrogate E (1-day forecast)",
    )


def _plot_timeline_infectious_7d_forecasts(
    step_trajectories: pd.DataFrame,
    figures_dir: Path,
    method_order: list[str],
) -> None:
    """Compare true infectious counts with aligned seven-day forecasts."""
    summaries: dict[str, pd.DataFrame] = {}
    for method in method_order:
        data = step_trajectories[step_trajectories["method"] == method]
        valid_forecasts = data.dropna(subset=["forecast_target_day_7d", "surrogate_I_7d"])
        true_summary = _trajectory_quantiles(data, "next_day", "true_I", "true")
        surrogate_summary = _trajectory_quantiles(
            valid_forecasts,
            "forecast_target_day_7d",
            "surrogate_I_7d",
            "surrogate",
        )
        summaries[method] = true_summary.join(surrogate_summary, how="inner")
    _plot_compartment_panels(
        summaries,
        figures_dir,
        method_order,
        ylabel="Infectious cases (I)",
        title="True infectious cases and 7-day surrogate forecasts",
        filename="infectious_7day_forecast_by_method.png",
        true_label="true I",
        surrogate_label="surrogate I (7-day forecast)",
    )


def plot_error_feature_ablation(summary: pd.DataFrame, path: str | Path) -> None:
    """Save cumulative error by K for each error-feature design."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, data in summary.groupby("state_design"):
        data = data.sort_values("error_history_length")
        ax.errorbar(
            data["error_history_length"],
            data["cumulative_error_mean"],
            yerr=data["cumulative_error_std"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=label,
        )
    ax.set_title("DQN cumulative error by error-state design")
    ax.set_xlabel("Error-history length K")
    ax.set_ylabel("Cumulative forecast error")
    ax.set_xticks(sorted(summary["error_history_length"].unique()))
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)


def plot_dqn_decision_surface(
    grid: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    """Save DQN decision-surface heatmaps from a precomputed state grid."""
    figures_dir = Path(figures_dir)
    plot_specs = [
        (
            "q_margin_recalibrate_minus_wait",
            "Q(recalibrate) - Q(wait)",
            "dqn_decision_surface_q_margin.png",
            "coolwarm",
        ),
        (
            "softmax_p_recalibrate",
            "Softmax probability of recalibration",
            "dqn_decision_surface_recalibration_probability.png",
            "viridis",
        ),
        (
            "greedy_action",
            "Greedy action: 1 = recalibrate",
            "dqn_decision_surface_greedy_action.png",
            "magma",
        ),
    ]

    patterns = list(grid["error_pattern"].drop_duplicates())
    for value_col, title, filename, cmap in plot_specs:
        fig, axes = plt.subplots(1, len(patterns), figsize=(5 * len(patterns), 4.4), sharey=True)
        if len(patterns) == 1:
            axes = [axes]

        vmin = grid[value_col].min()
        vmax = grid[value_col].max()
        if value_col == "q_margin_recalibrate_minus_wait":
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        elif value_col == "greedy_action":
            vmin, vmax = 0, 1

        image = None
        for ax, pattern in zip(axes, patterns):
            data = grid[grid["error_pattern"] == pattern]
            pivot = data.pivot(
                index="recent_error_level",
                columns="infectious_fraction",
                values=value_col,
            ).sort_index()
            x_values = pivot.columns.to_numpy(dtype=float)
            y_values = pivot.index.to_numpy(dtype=float)
            z_values = pivot.to_numpy(dtype=float)
            image = ax.imshow(
                z_values,
                origin="lower",
                aspect="auto",
                extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            contour_level = 0.5 if value_col in {"softmax_p_recalibrate", "greedy_action"} else 0.0
            if np.nanmin(z_values) < contour_level < np.nanmax(z_values):
                ax.contour(
                    x_values,
                    y_values,
                    z_values,
                    levels=[contour_level],
                    colors="white" if value_col != "greedy_action" else "black",
                    linewidths=1.0,
                    alpha=0.75,
                )
            ax.set_title(pattern.replace("_", " "))
            ax.set_xlabel("I / N")
            ax.set_ylabel("Recent forecast error level")

        fig.suptitle(title)
        if image is not None:
            fig.colorbar(image, ax=list(axes), location="right", shrink=0.86, pad=0.02)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_episode_diagnostics(trajectories: pd.DataFrame, path: str | Path, episode: int = 0) -> None:
    """Compatibility diagnostic panel for one episode."""
    data = _episode_data(trajectories, episode=episode)
    if data.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for policy, group in data.groupby("policy"):
        axes[0].plot(group["t"], group["forecast_error"], label=policy)
        axes[1].plot(group["t"], group["beta_hat"], label=policy)
        axes[2].step(group["t"], group["recalibrated"], where="post", label=policy)
    beta = data.groupby("t")["true_beta"].first()
    axes[1].plot(beta.index, beta.values, color="black", linestyle="--", label="true beta")
    axes[0].set_ylabel("Forecast error")
    axes[1].set_ylabel("Beta")
    axes[2].set_ylabel("Recalibrated")
    axes[2].set_xlabel("Day")
    axes[0].legend(loc="upper right", ncols=2)
    axes[1].legend(loc="upper right", ncols=2)
    _save(fig, path)


def plot_representative_episode_diagnostics(
    trajectories: pd.DataFrame,
    path: str | Path,
    episode: int = 0,
    policies: list[str] | None = None,
    surge_start: int | None = None,
    surge_end: int | None = None,
) -> None:
    """Save diagnostics for DQN and one representative baseline of each type."""
    if policies is None:
        policies = ["dqn", "threshold_0.003", "fixed_schedule_14"]
    data = trajectories[trajectories["policy"].isin(policies)]
    data = _episode_data(data, episode=episode)
    if data.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for policy, group in data.groupby("policy"):
        axes[0].plot(group["t"], group["forecast_error"], label=policy)
        axes[1].plot(group["t"], group["beta_hat"], label=policy)
        axes[2].step(group["t"], group["recalibrated"], where="post", label=policy)

    beta = data.groupby("t")["true_beta"].first()
    axes[1].plot(beta.index, beta.values, color="black", linestyle="--", label="true beta")
    for idx, ax in enumerate(axes):
        _mark_surge_period(ax, surge_start, surge_end, label=idx == 0)

    axes[0].set_title("Representative episode diagnostics")
    axes[0].set_ylabel("Forecast error")
    axes[1].set_ylabel("Beta")
    axes[2].set_ylabel("Recalibrated")
    axes[2].set_xlabel("Day")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].legend(loc="upper right", fontsize=8)
    _save(fig, path)


def plot_policy_errors(summary: pd.DataFrame, path: str | Path) -> None:
    """Compatibility wrapper for cumulative error comparisons."""
    plot_cumulative_error_comparison(summary, path)


def plot_dqn_recalibration_comparison(
    episode_summary: pd.DataFrame,
    trajectories: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    """Plot policy metrics for each beta recalibration estimator."""
    figures_dir = Path(figures_dir)
    policy_order = ["fixed_schedule", "error_threshold", "dqn"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    metrics = [
        ("cumulative_error", "Cumulative one-step error", "cumulative_one_step_error.png"),
        ("mean_path_error", "Mean seven-day path error", "seven_day_path_error.png"),
        ("mean_mixed_error", "Mean causal mixed error", "causal_mixed_error.png"),
        ("budget_used", "Budget used", "budget_used.png"),
    ]
    methods = list(episode_summary["recalibration_method_group"].drop_duplicates())
    for metric, ylabel, filename in metrics:
        fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, method in zip(axes, methods):
            data = episode_summary[episode_summary["recalibration_method_group"] == method]
            values = [data.loc[data["policy"] == policy, metric].dropna().to_numpy() for policy in policy_order]
            boxes = ax.boxplot(
                values, tick_labels=policy_order, patch_artist=True, showfliers=False
            )
            for box, color in zip(boxes["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(0.78)
            ax.set_title(method.replace("_", " "))
            ax.set_xlabel("Policy")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"{ylabel} by policy and recalibration method")
        _save(fig, figures_dir / filename, layout_kwargs={"rect": (0, 0, 1, 0.95)})

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.arange(0, 171, 10)
    for ax, method in zip(axes, methods):
        method_data = trajectories[trajectories["recalibration_method_group"] == method]
        for policy, color in zip(policy_order, colors):
            all_policy_data = method_data[method_data["policy"] == policy]
            times = all_policy_data.loc[all_policy_data["recalibrated"] == 1, "decision_t"]
            episode_count = len(
                all_policy_data[["training_seed", "episode"]].drop_duplicates()
            )
            weights = np.full(len(times), 1.0 / max(episode_count, 1))
            ax.hist(
                times,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=2,
                color=color,
                label=policy,
            )
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("Recalibration day")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean recalibrations per episode")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Recalibration timing")
    _save(fig, figures_dir / "recalibration_timing.png", layout_kwargs={"rect": (0, 0, 1, 0.95)})

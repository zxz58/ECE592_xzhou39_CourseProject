"""Evaluation metrics for recalibration experiments."""

from __future__ import annotations

import pandas as pd


def summarize_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Summarize trajectory rows into one row per policy and episode."""
    grouped = trajectories.groupby(["policy", "episode"], as_index=False)
    summary = grouped.agg(
        cumulative_error=("forecast_error", "sum"),
        average_forecast_error=("forecast_error", "mean"),
        episode_length=("t", "max"),
        budget_allowed=("initial_budget", "last"),
        budget_used=("budget_used", "last"),
        budget_utilization=("budget_utilization", "last"),
        number_of_recalibrations=("recalibrated", "sum"),
        termination_reason=("termination_reason", "last"),
        final_active_infections=("active_infections", "last"),
        final_beta_hat=("beta_hat", "last"),
        peak_true_I=("true_I", "max"),
        final_S=("true_S", "last"),
        final_E=("true_E", "last"),
        final_I=("true_I", "last"),
        final_R=("true_R", "last"),
    )
    times = (
        trajectories[trajectories["recalibrated"] == 1]
        .groupby(["policy", "episode"])["t"]
        .apply(lambda values: ";".join(str(int(value)) for value in values))
        .reset_index(name="recalibration_times")
    )
    summary = summary.merge(times, on=["policy", "episode"], how="left")
    summary["recalibration_times"] = summary["recalibration_times"].fillna("")
    summary["final_epidemic_size"] = summary["final_E"] + summary["final_I"] + summary["final_R"]
    return summary


def compare_policies(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode summaries into policy comparison metrics."""
    grouped = summary.groupby("policy", as_index=False)
    return grouped.agg(
        cumulative_error_mean=("cumulative_error", "mean"),
        cumulative_error_std=("cumulative_error", "std"),
        average_forecast_error_mean=("average_forecast_error", "mean"),
        average_forecast_error_std=("average_forecast_error", "std"),
        episode_length_mean=("episode_length", "mean"),
        episode_length_std=("episode_length", "std"),
        budget_allowed_mean=("budget_allowed", "mean"),
        budget_used_mean=("budget_used", "mean"),
        budget_used_std=("budget_used", "std"),
        budget_utilization_mean=("budget_utilization", "mean"),
        budget_utilization_std=("budget_utilization", "std"),
        number_of_recalibrations_mean=("number_of_recalibrations", "mean"),
        number_of_recalibrations_std=("number_of_recalibrations", "std"),
        final_active_infections_mean=("final_active_infections", "mean"),
        final_epidemic_size_mean=("final_epidemic_size", "mean"),
        final_epidemic_size_std=("final_epidemic_size", "std"),
        peak_true_I_mean=("peak_true_I", "mean"),
    )

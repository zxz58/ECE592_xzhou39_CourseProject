"""Threshold-based recalibration baseline."""


class ThresholdPolicy:
    """Recalibrate when the latest normalized forecast error is high."""

    name = "threshold"

    def __init__(
        self,
        error_threshold: float = 0.003,
        use_trend: bool = False,
        trend_threshold: float | None = None,
        error_start_index: int = 4,
        error_history_length: int = 7,
        cooldown: int = 0,
        threshold: float | None = None,
    ) -> None:
        self.error_threshold = float(error_threshold if threshold is None else threshold)
        self.use_trend = bool(use_trend)
        self.trend_threshold = trend_threshold
        self.error_start_index = int(error_start_index)
        self.error_history_length = int(error_history_length)
        self.cooldown = int(cooldown)
        self.name = f"threshold_{self.error_threshold:g}"

    def reset(self) -> None:
        """No internal state is needed."""

    def action(self, state, info: dict) -> int:
        """Recalibrate when recent error or error trend exceeds threshold."""
        remaining_budget = int(info.get("remaining_budget", 1))
        days_since = int(info.get("days_since_recalibration", self.cooldown))
        if remaining_budget <= 0 or days_since < self.cooldown:
            return 0

        rolling_error = float(info.get("rolling_mean_forecast_error", info.get("error", 0.0)))
        if rolling_error >= self.error_threshold:
            return 1

        if self.use_trend:
            threshold = self.trend_threshold
            if threshold is None:
                threshold = self.error_threshold / max(self.error_history_length, 1)
            trend = float(info.get("rolling_error_trend", 0.0))
            return int(trend >= threshold)
        return 0

    def act(self, observation) -> int:
        """Compatibility wrapper for older experiment code."""
        return self.action(observation, {"remaining_budget": 1})

"""Fixed-schedule recalibration baseline."""


class FixedSchedulePolicy:
    """Recalibrate every fixed number of days while budget remains."""

    name = "fixed_schedule"

    def __init__(self, interval: int = 14, start_day: int = 0, cooldown: int = 0) -> None:
        self.interval = int(interval)
        self.start_day = int(start_day)
        self.cooldown = int(cooldown)
        self.name = f"fixed_schedule_{self.interval}"
        self.step_count = 0

    def reset(self) -> None:
        """Restart the policy clock."""
        self.step_count = 0

    def action(self, state, info: dict) -> int:
        """Return 1 every k days when budget remains, otherwise 0."""
        self.step_count += 1
        remaining_budget = int(info.get("remaining_budget", 1))
        t = int(info.get("t", self.step_count))
        days_since = int(info.get("days_since_recalibration", self.cooldown))
        if remaining_budget <= 0 or t < self.start_day or days_since < self.cooldown:
            return 0
        return int((t - self.start_day) % max(self.interval, 1) == 0)

    def act(self, observation) -> int:
        """Compatibility wrapper for older experiment code."""
        return self.action(observation, {"remaining_budget": 1})


class FixedDaysPolicy:
    """Recalibrate at explicit simulator times while budget remains."""

    def __init__(self, trigger_days: list[int]) -> None:
        self.trigger_days = frozenset(int(day) for day in trigger_days)
        if any(day < 0 for day in self.trigger_days):
            raise ValueError("trigger days must be nonnegative")
        self.name = "fixed_schedule"

    def reset(self) -> None:
        """The policy has no episode-local state."""

    def action(self, state, info: dict) -> int:
        """Request recalibration exactly at a configured day."""
        if int(info.get("remaining_budget", 1)) <= 0:
            return 0
        return int(int(info.get("t", -1)) in self.trigger_days)

    def act(self, observation) -> int:
        """Compatibility wrapper for older experiment code."""
        return 0

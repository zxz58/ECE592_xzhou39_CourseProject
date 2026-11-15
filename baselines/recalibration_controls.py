"""Simple recalibration control policies for diagnostic comparisons."""


class NeverRecalibratePolicy:
    """Always wait and preserve the initial surrogate beta."""

    name = "never_recalibrate"

    def reset(self) -> None:
        """No internal state is used."""

    def action(self, state, info: dict) -> int:
        """Always choose WAIT."""
        return 0

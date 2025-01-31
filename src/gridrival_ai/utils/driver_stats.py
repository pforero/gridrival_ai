from dataclasses import dataclass


@dataclass
class DriverStats:
    """Historical driver statistics needed for scoring.

    Parameters
    ----------
    rolling_averages : Dict[str, float]
        Mapping of driver_id to 8-race rolling average finish position
    """

    rolling_averages: dict[str, float]

    def __post_init__(self):
        """Validate rolling averages."""
        for driver_id, avg in self.rolling_averages.items():
            if not 1.0 <= avg <= 20.0:
                raise ValueError(f"Invalid average for {driver_id}: {avg}")

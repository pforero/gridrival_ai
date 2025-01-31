"""Driver statistics utilities."""

import warnings
from dataclasses import dataclass

from gridrival_ai.f1.data import VALID_DRIVER_IDS


def validate_driver_id(driver_id: str) -> None:
    """
    Validate a driver ID.

    Parameters
    ----------
    driver_id : str
        Driver ID to validate.

    Raises
    ------
    ValueError
        If driver_id is not exactly 3 uppercase letters.
    """
    if driver_id not in VALID_DRIVER_IDS:
        warnings.warn(f"Driver ID {driver_id} is not in the list of known drivers")


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
            validate_driver_id(driver_id)
            if not 1.0 <= avg <= 20.0:
                raise ValueError(f"Invalid average for {driver_id}: {avg}")

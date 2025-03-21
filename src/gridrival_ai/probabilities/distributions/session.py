"""
Session distribution class for representing probabilities of all drivers in a session.

This module provides the SessionDistribution class for modeling probability
distributions of all drivers in a specific racing session (qualifying, race, or sprint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from gridrival_ai.probabilities.distributions.joint import JointDistribution
from gridrival_ai.probabilities.distributions.position import PositionDistribution
from gridrival_ai.probabilities.normalizers.factory import get_grid_normalizer

# Constants
TOLERANCE = 1e-6


@dataclass(frozen=True)
class SessionDistribution:
    """
    Probability distribution for all drivers in a specific session.

    This class holds position distributions for all drivers in a session,
    provides checks for consistency, and methods for querying probabilities.

    Parameters
    ----------
    driver_distributions : Dict[str, PositionDistribution]
        Dictionary mapping driver IDs to their position distributions
    session_type : str
        Type of session ('race', 'qualifying', 'sprint')
    validate : bool, optional
        Whether to validate distributions on initialization, by default True

    Raises
    ------
    ValueError
        If validation fails and validate=True

    Examples
    --------
    >>> # Create position distributions for drivers
    >>> ver_dist = PositionDistribution({1: 0.6, 2: 0.3, 3: 0.1})
    >>> ham_dist = PositionDistribution({1: 0.3, 2: 0.5, 3: 0.2})
    >>>
    >>> # Create session distribution
    >>> session = SessionDistribution(
    ...     {"VER": ver_dist, "HAM": ham_dist},
    ...     session_type="race"
    ... )
    >>>
    >>> # Get distribution for a driver
    >>> session.get_driver_distribution("VER")[1]
    0.6
    >>>
    >>> # Get probabilities for a position
    >>> session.get_position_probabilities(1)
    {'VER': 0.6, 'HAM': 0.3}
    """

    driver_distributions: Dict[str, PositionDistribution]
    session_type: str
    _validate: bool = field(default=True, repr=False)

    VALID_SESSION_TYPES = ["race", "qualifying", "sprint"]

    def __post_init__(self) -> None:
        """Validate distributions after initialization."""
        if self._validate:
            self.validate()

    def validate(self) -> None:
        """
        Validate session distributions.

        Checks:
        - Session type is valid
        - All distributions are valid
        - At least one driver is present
        - All driver distributions have the same positions
        - The maximum position matches the number of drivers

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check session type
        if self.session_type not in self.VALID_SESSION_TYPES:
            raise ValueError(
                f"Invalid session type: {self.session_type}. "
                f"Valid types: {self.VALID_SESSION_TYPES}"
            )

        # Check at least one driver
        if not self.driver_distributions:
            raise ValueError("Must have at least one driver distribution")

        # Validate each distribution
        for driver_id, dist in self.driver_distributions.items():
            if not isinstance(dist, PositionDistribution):
                raise ValueError(
                    f"Distribution for driver {driver_id} must be a "
                    "PositionDistribution"
                )

            # Check if distribution is valid
            if not dist.is_valid:
                raise ValueError(f"Invalid distribution for driver {driver_id}")

        # Check that all drivers have the same positions
        driver_ids = list(self.driver_distributions.keys())
        positions_sets = []

        for driver_id in driver_ids:
            positions = set(self.driver_distributions[driver_id].position_probs.keys())
            positions_sets.append(positions)

        # Get the first positions set as a reference
        if positions_sets:
            reference_positions = positions_sets[0]

            # Check that all other positions sets match the reference
            for i, positions in enumerate(positions_sets[1:], 1):
                if positions != reference_positions:
                    raise ValueError(
                        f"Driver {driver_ids[i]} has different positions "
                        f"than driver {driver_ids[0]}"
                    )

        # Check that the maximum position matches the number of drivers
        if positions_sets:
            max_position = max(positions_sets[0])
            if max_position != len(self.driver_distributions):
                raise ValueError(
                    f"Maximum position ({max_position}) does not match "
                    f"number of drivers ({len(self.driver_distributions)})"
                )

    def get_driver_distribution(self, driver_id: str) -> PositionDistribution:
        """
        Get position distribution for a specific driver.

        Parameters
        ----------
        driver_id : str
            Driver ID

        Returns
        -------
        PositionDistribution
            Position distribution for the driver

        Raises
        ------
        KeyError
            If driver not found
        """
        if driver_id not in self.driver_distributions:
            raise KeyError(f"Driver {driver_id} not found in session distribution")
        return self.driver_distributions[driver_id]

    def get_position_probabilities(self, position: int) -> Dict[str, float]:
        """
        Get probabilities of all drivers finishing in a specific position.

        Parameters
        ----------
        position : int
            Position to query

        Returns
        -------
        Dict[str, float]
            Dictionary mapping driver IDs to probabilities
        """
        return {
            driver_id: dist.get(position)
            for driver_id, dist in self.driver_distributions.items()
        }

    def get_driver_ids(self) -> List[str]:
        """
        Get all driver IDs in the session.

        Returns
        -------
        List[str]
            List of driver IDs
        """
        return list(self.driver_distributions.keys())

    def get_positions(self) -> List[int]:
        """
        Get all positions in the session distributions.

        Returns
        -------
        List[int]
            List of positions
        """
        if not self.driver_distributions:
            return []

        # Since all drivers should have the same positions, we can use any driver
        first_driver_id = next(iter(self.driver_distributions.keys()))
        positions = list(
            self.driver_distributions[first_driver_id].position_probs.keys()
        )
        return sorted(positions)

    def get_joint_distribution(
        self, driver1_id: str, driver2_id: str
    ) -> JointDistribution:
        """
        Get joint distribution between two drivers.

        Parameters
        ----------
        driver1_id : str
            First driver ID
        driver2_id : str
            Second driver ID

        Returns
        -------
        JointDistribution
            Joint distribution between the two drivers

        Raises
        ------
        KeyError
            If either driver not found
        """
        dist1 = self.get_driver_distribution(driver1_id)
        dist2 = self.get_driver_distribution(driver2_id)

        # In race, two drivers can't have the same position
        constrained = self.session_type == "race"

        return JointDistribution.create_from_distributions(
            dist1, dist2, driver1_id, driver2_id, constrained
        )

    def normalize(self, method: str = "sinkhorn", **kwargs) -> "SessionDistribution":
        """
        Normalize position probabilities to ensure consistency.

        Parameters
        ----------
        method : str, optional
            Normalization method, by default "sinkhorn"
        **kwargs
            Additional parameters for the normalizer

        Returns
        -------
        SessionDistribution
            Normalized session distribution
        """
        # Get normalizer
        normalizer = get_grid_normalizer(method, **kwargs)

        # Normalize distributions
        normalized_dists = normalizer.normalize(self.driver_distributions)

        # Create new session distribution
        return SessionDistribution(normalized_dists, self.session_type, _validate=False)

    def get_most_likely_grid(self) -> Dict[int, str]:
        """
        Get the most likely grid configuration.

        Returns
        -------
        Dict[int, str]
            Dictionary mapping positions to driver IDs
        """
        # For each position, find driver with highest probability
        grid = {}
        assigned_drivers = set()

        # Get all positions
        positions = self.get_positions()

        # Greedy assignment - works well for race where constrained
        for pos in positions:
            # Get unassigned drivers with probabilities for this position
            candidates = {
                driver_id: dist.get(pos)
                for driver_id, dist in self.driver_distributions.items()
                if driver_id not in assigned_drivers
            }

            # Skip if no candidates
            if not candidates:
                continue

            # Assign driver with highest probability
            best_driver = max(candidates.items(), key=lambda x: x[1])[0]
            grid[pos] = best_driver
            assigned_drivers.add(best_driver)

        return grid

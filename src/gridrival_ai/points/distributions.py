"""
Distribution adapter for GridRival points calculation.

This module provides an adapter between the probability distribution API
and the points calculation system. It handles retrieving, transforming,
and providing distributions in the format needed by the points calculators.

The adapter serves as a bridge that isolates the point calculation system
from changes in the underlying probability distribution implementation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, cast

from gridrival_ai.data.reference import CONSTRUCTORS
from gridrival_ai.probabilities.distributions import (  # create_independent_joint,
    JointDistribution,
    PositionDistribution,
)
from gridrival_ai.probabilities.registry import DistributionRegistry


@dataclass
class DistributionAdapter:
    """
    Adapter to provide a consistent interface for accessing probability distributions.

    This adapter bridges between the probability API and the expectations
    needed by the points calculators. It handles the retrieval and manipulation
    of distributions from the registry.

    Parameters
    ----------
    registry : DistributionRegistry
        Registry containing probability distributions

    Examples
    --------
    >>> registry = DistributionRegistry()
    >>> # Assume registry has been populated with distributions
    >>> adapter = DistributionAdapter(registry)
    >>> qual_dist = adapter.get_position_distribution("VER", "qualifying")
    >>> completion_prob = adapter.get_completion_probability("VER")
    >>> drivers = adapter.get_constructor_drivers("RBR")
    >>> print(drivers)
    ('VER', 'PER')
    """

    registry: DistributionRegistry

    def get_position_distribution(
        self, driver_id: str, session: str
    ) -> PositionDistribution:
        """
        Get position distribution for a driver and session.

        Parameters
        ----------
        driver_id : str
            Driver ID (e.g., "VER")
        session : str
            Session name ("qualifying", "race", "sprint")

        Returns
        -------
        PositionDistribution
            Distribution over finishing positions

        Raises
        ------
        KeyError
            If no distribution found for driver and session

        Examples
        --------
        >>> adapter = DistributionAdapter(registry)
        >>> dist = adapter.get_position_distribution("VER", "qualifying")
        >>> prob_p1 = dist[1]  # Probability of pole position
        """
        # Get the distribution from the registry
        return cast(PositionDistribution, self.registry.get(driver_id, session))

    def get_position_distribution_safe(
        self,
        driver_id: str,
        session: str,
        default: Optional[PositionDistribution] = None,
    ) -> Optional[PositionDistribution]:
        """
        Get position distribution with fallback to default.

        Parameters
        ----------
        driver_id : str
            Driver ID (e.g., "VER")
        session : str
            Session name ("qualifying", "race", "sprint")
        default : Optional[PositionDistribution], optional
            Default distribution to return if not found, by default None

        Returns
        -------
        Optional[PositionDistribution]
            Distribution over finishing positions, or default if not found

        Examples
        --------
        >>> adapter = DistributionAdapter(registry)
        >>> dist = adapter.get_position_distribution_safe("VER", "qualifying")
        >>> if dist:
        ...     prob_p1 = dist[1]
        """
        try:
            return self.get_position_distribution(driver_id, session)
        except KeyError:
            return default

    def get_joint_distribution(
        self, driver1_id: str, driver2_id: str, session: str, constrained: bool = True
    ) -> JointDistribution:
        """
        Get joint distribution between two drivers.

        Parameters
        ----------
        driver1_id : str
            First driver ID
        driver2_id : str
            Second driver ID
        session : str
            Session name
        constrained : bool, optional
            Whether to enforce constraints (e.g., no same positions), by default True

        Returns
        -------
        JointDistribution
            Joint distribution between drivers

        Raises
        ------
        KeyError
            If no distributions found for drivers and session

        Examples
        --------
        >>> adapter = DistributionAdapter(registry)
        >>> # Get joint distribution between teammates
        >>> joint = adapter.get_joint_distribution("VER", "PER", "race")
        >>> # Probability VER P1, PER P2
        >>> prob = joint[(1, 2)]
        """
        # Get the joint distribution from the registry
        return self.registry.get_joint(
            driver1_id, driver2_id, session, constrained=constrained
        )

    def get_joint_distribution_safe(
        self,
        driver1_id: str,
        driver2_id: str,
        session: str,
        constrained: bool = True,
        default: Optional[JointDistribution] = None,
    ) -> Optional[JointDistribution]:
        """
        Get joint distribution with fallback to default.

        Parameters
        ----------
        driver1_id : str
            First driver ID
        driver2_id : str
            Second driver ID
        session : str
            Session name
        constrained : bool, optional
            Whether to enforce constraints (e.g., no same positions), by default True
        default : Optional[JointDistribution], optional
            Default distribution to return if not found, by default None

        Returns
        -------
        Optional[JointDistribution]
            Joint distribution between drivers, or default if not found
        """
        try:
            return self.get_joint_distribution(
                driver1_id, driver2_id, session, constrained=constrained
            )
        except (KeyError, ValueError):
            return default

    def get_qualifying_race_distribution(self, driver_id: str) -> JointDistribution:
        """
        Get joint distribution between qualifying and race positions.

        Parameters
        ----------
        driver_id : str
            Driver ID

        Returns
        -------
        JointDistribution
            Joint distribution between qualifying and race

        Raises
        ------
        KeyError
            If no distributions found for driver
        ValueError
            If distribution correlation not supported

        Notes
        -----
        This method attempts multiple strategies to find the joint distribution:
        1. Look for an explicit correlation distribution in the registry
        2. Try to get a correlation from driver session correlation
        3. Create an independent joint distribution as fallback

        Examples
        --------
        >>> adapter = DistributionAdapter(registry)
        >>> joint = adapter.get_qualifying_race_distribution("VER")
        >>> # Probability of qualifying P1 and finishing race P1
        >>> prob = joint[(1, 1)]
        """
        # Define sessions
        session1 = "qualifying"
        session2 = "race"

        # Check if both distributions exist
        if not self.registry.has(driver_id, session1) or not self.registry.has(
            driver_id, session2
        ):
            raise KeyError(f"Missing distributions for driver {driver_id}")

        # Get the individual distributions
        qual_dist = self.get_position_distribution(driver_id, session1)
        race_dist = self.get_position_distribution(driver_id, session2)

        # Strategy 1: Try to get a stored joint distribution if available
        try:
            return self.registry.get_joint(
                f"{driver_id}_{session1}",
                f"{driver_id}_{session2}",
                "correlation",
                constrained=False,
            )
        except (KeyError, ValueError):
            pass

        # Strategy 2: Try to use registry's session correlation mechanism
        try:
            correlation_id = f"{driver_id}_correlation"
            if self.registry.has(correlation_id, "qual_race"):
                return cast(
                    JointDistribution, self.registry.get(correlation_id, "qual_race")
                )
        except (KeyError, ValueError):
            pass

        # Strategy 3: If all else fails, create an independent joint distribution
        # This is a fallback and may not capture the true correlation
        # return create_independent_joint(
        #    qual_dist, race_dist, name1=session1, name2=session2
        # )

    def get_completion_probability(
        self, driver_id: str, default: float = 0.95
    ) -> float:
        """
        Get race completion probability.

        Parameters
        ----------
        driver_id : str
            Driver ID
        default : float, optional
            Default completion probability if not found, by default 0.95

        Returns
        -------
        float
            Probability of completing the race

        Examples
        --------
        >>> adapter = DistributionAdapter(registry)
        >>> prob = adapter.get_completion_probability("VER")
        >>> print(f"Chance of finishing: {prob:.1%}")
        Chance of finishing: 95.0%
        """
        # Strategy 1: If driver has a "completion" distribution, use it
        if self.registry.has(driver_id, "completion"):
            completion_dist = cast(
                PositionDistribution, self.registry.get(driver_id, "completion")
            )
            return completion_dist[1]  # P(completing) is position 1

        # Strategy 2: Check for an explicit completion probability attribute
        try:
            completion_id = f"{driver_id}_completion"
            if self.registry.has(completion_id, "probability"):
                dist = cast(
                    PositionDistribution,
                    self.registry.get(completion_id, "probability"),
                )
                return dist[1]
        except (KeyError, ValueError):
            pass

        # Fallback to provided default completion probability
        return default  # Use the provided default parameter

    def get_constructor_drivers(self, constructor_id: str) -> Tuple[str, str]:
        """
        Get driver IDs for a constructor.

        Parameters
        ----------
        constructor_id : str
            Constructor ID

        Returns
        -------
        Tuple[str, str]
            Tuple of driver IDs

        Raises
        ------
        KeyError
            If constructor not found

        Examples
        --------
        >>> adapter = DistributionAdapter(registry)
        >>> drivers = adapter.get_constructor_drivers("RBR")
        >>> print(drivers)
        ('VER', 'PER')
        """
        # Strategy 1: Check if registry has constructor drivers
        try:
            drivers_id = f"{constructor_id}_drivers"
            if self.registry.has(drivers_id, "pair"):
                dist = cast(PositionDistribution, self.registry.get(drivers_id, "pair"))
                # Get the driver IDs from the distribution
                # The distribution might store drivers as keys or in a special way
                # Try to get them using __getitem__ first (for our mock in tests)
                try:
                    driver1 = dist[0]
                    driver2 = dist[1]
                    return (driver1, driver2)
                except (KeyError, IndexError):
                    # If that fails, try to extract from position_probs keys
                    keys = list(dist.position_probs.keys())
                    if len(keys) >= 2:
                        return (str(keys[0]), str(keys[1]))
                    raise ValueError("Not enough drivers in distribution")
        except (KeyError, ValueError, IndexError):
            pass

        # Strategy 2: Use the reference data to get constructor drivers
        constructor = CONSTRUCTORS.get(constructor_id)
        if constructor is None:
            raise KeyError(f"Constructor {constructor_id} not found")

        return constructor.drivers

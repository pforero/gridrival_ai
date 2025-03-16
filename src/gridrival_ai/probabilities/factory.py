"""
Factory for creating probability distributions from betting odds.

This module provides factory methods for creating F1 race probability
distributions from structured betting odds data. It simplifies the process
of converting market odds into position distributions for drivers across
different race sessions.
"""

from typing import Dict, List, Optional, Set

from gridrival_ai.probabilities.conversion import ConverterFactory
from gridrival_ai.probabilities.core import PositionDistribution
from gridrival_ai.probabilities.registry import DistributionRegistry


class DistributionFactory:
    """
    Factory for creating race probability distributions from betting odds.

    This class provides methods for converting structured betting odds
    into position distributions for drivers and sessions, as well as
    registering these distributions with a distribution registry.

    Examples
    --------
    >>> # Define structured odds
    >>> odds_structure = {
    ...     "race": {
    ...         1: {"VER": 5, "HAM": 7.75, "NOR": 3},  # Win odds
    ...         3: {"VER": 2, "HAM": 3, "NOR": 1.5},   # Top 3 odds
    ...     },
    ...     "qualification": {
    ...         1: {"VER": 3.5, "HAM": 6, "NOR": 2.5}, # Pole position odds
    ...     }
    ... }
    >>>
    >>> # Create distributions for all drivers and sessions
    >>> distributions = DistributionFactory.from_structured_odds(odds_structure)
    >>>
    >>> # Access specific driver distribution
    >>> ver_race_dist = distributions["race"]["VER"]
    >>> print(f"Probability of VER finishing P1: {ver_race_dist[1]:.1%}")
    """

    @staticmethod
    def from_structured_odds(
        odds_structure: Dict[str, Dict[int, Dict[str, float]]],
        method: str = "basic",
        fallback_to_race: bool = True,
        **kwargs,
    ) -> Dict[str, Dict[str, PositionDistribution]]:
        """
        Create position distributions from a structured betting odds dictionary.

        Parameters
        ----------
        odds_structure : Dict[str, Dict[int, Dict[str, float]]]
            Nested dictionary with format:
            {
                "race": {
                    1: {"VER": 5, "HAM": 7.75, ...},  # Win odds
                    3: {"VER": 2, "HAM": 3.5, ...},   # Top 3 odds
                    ...
                },
                "qualification": {...},
                "sprint": {...}
            }
        method : str, optional
            Conversion method name, by default "basic"
            Options: "basic", "odds_ratio", "shin", "power", "harville"
        fallback_to_race : bool, optional
            Whether to use race odds when qualification or sprint odds are missing,
            by default True
        **kwargs
            Additional parameters for the converter

        Returns
        -------
        Dict[str, Dict[str, PositionDistribution]]
            Dictionary mapping race types to dictionaries of driver distributions:
            {
                "race": {"VER": PositionDistribution, "HAM": PositionDistribution, ...},
                "qualification": {...},
                "sprint": {...}
            }

        Notes
        -----
        The position keys (1, 3, 6, etc.) represent finish position thresholds.
        For example, position 3 means odds of finishing in positions 1, 2, or 3.
        Multiple position thresholds are used to create more accurate distributions.
        """
        result = {}

        # Collect all driver IDs across all sessions
        all_drivers: Set[str] = set()
        for session in odds_structure:
            for position in odds_structure[session]:
                all_drivers.update(odds_structure[session][position].keys())

        # Get race data for fallback if needed
        race_data = odds_structure.get("race", {})

        # Only process sessions that exist in the input structure or are explicitly
        # requested
        sessions_to_process = []

        # First add sessions that exist in the input
        for session in odds_structure.keys():
            sessions_to_process.append(session)

        # Then consider fallback if enabled
        if fallback_to_race and "race" in odds_structure:
            for fallback_session in ["qualification", "sprint"]:
                # Only add fallback sessions if they don't already exist
                if fallback_session not in sessions_to_process:
                    sessions_to_process.append(fallback_session)

        # Process each session
        for session in sessions_to_process:
            # Get session data from the odds structure if it exists
            if session in odds_structure:
                session_data = odds_structure[session]
                # If data is empty and fallback is enabled, use race data instead
                if (
                    not session_data
                    and fallback_to_race
                    and session in ["qualification", "sprint"]
                    and "race" in odds_structure
                ):
                    session_data = race_data
            elif (
                fallback_to_race
                and session in ["qualification", "sprint"]
                and "race" in odds_structure
            ):
                session_data = race_data
            else:
                continue

            # Skip if still no data
            if not session_data:
                continue

            # Process each threshold position
            threshold_probs: Dict[int, Dict[str, float]] = {}
            for position, odds_dict in session_data.items():
                # Each position threshold should sum to the position value
                # (e.g., for position 3, probabilities should sum to 3.0)
                threshold_probs[position] = {}

                # Collect valid odds for all drivers
                valid_odds = {}
                valid_driver_ids = []
                for driver_id, odds in odds_dict.items():
                    if odds <= 1.0:
                        continue  # Skip invalid odds
                    valid_odds[driver_id] = odds
                    valid_driver_ids.append(driver_id)

                if not valid_odds:
                    continue  # Skip if no valid odds

                # Convert odds to probabilities using the specified method
                converter = ConverterFactory.get(method, **kwargs)
                odds_list = [valid_odds[d] for d in valid_driver_ids]
                probs = converter.convert(odds_list, target_sum=float(position))

                # Store probabilities for each driver
                for i, driver_id in enumerate(valid_driver_ids):
                    threshold_probs[position][driver_id] = probs[i]

            # Convert threshold probabilities to position distributions
            driver_dists = DistributionFactory._create_position_distributions(
                threshold_probs, all_drivers, **kwargs
            )

            result[session] = driver_dists

        return result

    @staticmethod
    def _create_position_distributions(
        threshold_probs: Dict[int, Dict[str, float]], all_drivers: Set[str], **kwargs
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions from threshold probabilities.

        Parameters
        ----------
        threshold_probs : Dict[int, Dict[str, float]]
            Dictionary mapping threshold positions to driver probabilities
        all_drivers : Set[str]
            Set of all driver IDs across all sessions
        **kwargs
            Additional parameters for converter

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        driver_dists = {}

        # Maximum position we'll consider (can be adjusted as needed)
        max_position = len(all_drivers)

        # Get sorted thresholds
        thresholds = sorted([t for t in threshold_probs.keys() if t <= max_position])
        if not thresholds:
            return {}

        # Step 1: Extract cumulative probabilities for each driver at each threshold
        # This maps driver_id -> position -> cumulative probability
        driver_cum_probs = {driver_id: {} for driver_id in all_drivers}

        for threshold in thresholds:
            threshold_data = threshold_probs.get(threshold, {})
            for driver_id in all_drivers:
                # Get probability for this driver at this threshold
                cum_prob = threshold_data.get(driver_id, 0.0)
                driver_cum_probs[driver_id][threshold] = cum_prob

        # Step 2: Convert cumulative probabilities to position-specific probabilities
        # This maps driver_id -> position -> exact position probability
        driver_exact_probs = {driver_id: {} for driver_id in all_drivers}

        for driver_id in all_drivers:
            cum_probs = driver_cum_probs[driver_id]
            last_cum_prob = 0.0

            # Process each threshold to get position-specific probabilities
            for i, threshold in enumerate(thresholds):
                current_cum_prob = cum_probs.get(threshold, 0.0)

                # Skip if probability does not increase
                if current_cum_prob <= last_cum_prob:
                    continue

                # Calculate exact probability for this range
                range_prob = current_cum_prob - last_cum_prob

                # Determine position range this threshold represents
                start_pos = 1 if i == 0 else thresholds[i - 1] + 1
                end_pos = threshold

                # Calculate positions in this range
                positions_in_range = list(range(start_pos, end_pos + 1))
                if not positions_in_range:
                    continue

                # Distribute probability across positions in this range
                # We use a weighted distribution where earlier positions get higher
                # probability. This better reflects reality where P1 is harder than P2,
                # P2 harder than P3, etc.
                total_weight = sum(1 / p for p in positions_in_range)
                for pos in positions_in_range:
                    # Weight is inversely proportional to position
                    weight = (1 / pos) / total_weight
                    exact_prob = range_prob * weight
                    driver_exact_probs[driver_id][pos] = exact_prob

                last_cum_prob = current_cum_prob

        # Step 3: Fill in missing positions with small probabilities to ensure all
        # drivers have a full distribution
        for driver_id in all_drivers:
            for pos in range(1, max_position + 1):
                if pos not in driver_exact_probs[driver_id]:
                    # Add small probability for missing positions
                    driver_exact_probs[driver_id][pos] = 1e-6

        # Step 4: Ensure all position probabilities sum to 1.0 for each driver
        for driver_id in all_drivers:
            probs = driver_exact_probs[driver_id]
            if not probs:
                continue

            # Normalize to ensure sum to 1.0
            total = sum(probs.values())
            if total > 0:
                normalized = {k: v / total for k, v in probs.items()}
                try:
                    driver_dists[driver_id] = PositionDistribution(normalized)
                except Exception:
                    # If validation fails, try without validation and normalize
                    dist = PositionDistribution(normalized, _validate=False)
                    driver_dists[driver_id] = dist.normalize()

        return driver_dists

    @staticmethod
    def register_structured_odds(
        registry: DistributionRegistry,
        odds_structure: Dict[str, Dict[int, Dict[str, float]]],
        method: str = "basic",
        fallback_to_race: bool = True,
        **kwargs,
    ) -> None:
        """
        Register distributions from structured odds with a distribution registry.

        Parameters
        ----------
        registry : DistributionRegistry
            Registry to register distributions with
        odds_structure : Dict[str, Dict[int, Dict[str, float]]]
            Structured odds dictionary (see from_structured_odds for format)
        method : str, optional
            Conversion method, by default "basic"
        fallback_to_race : bool, optional
            Whether to use race odds when qualification or sprint odds are missing,
            by default True
        **kwargs
            Additional parameters for converter

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> DistributionFactory.register_structured_odds(registry, odds_structure)
        >>> race_dist = registry.get("VER", "race")
        """
        distributions = DistributionFactory.from_structured_odds(
            odds_structure, method=method, fallback_to_race=fallback_to_race, **kwargs
        )

        # Register each distribution with the registry
        # Only register distributions for sessions that:
        # 1. Exist in the input structure, OR
        # 2. Are qualification/sprint specifically requested via fallback
        for session, driver_dists in distributions.items():
            should_register = False

            # Case 1: The session exists in the original odds structure
            if session in odds_structure:
                should_register = True
            # Case 2: Session is a fallback session AND fallback is enabled AND we have
            # race data
            elif (
                session in ["qualification", "sprint"]
                and fallback_to_race
                and "race" in odds_structure
            ):
                should_register = True

            # Only register if we should
            if should_register:
                for driver_id, dist in driver_dists.items():
                    registry.register(driver_id, session, dist)

    @staticmethod
    def from_odds_dict(
        odds_dict: Dict[str, float], method: str = "basic", **kwargs
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions from a dictionary of driver odds.

        Parameters
        ----------
        odds_dict : Dict[str, float]
            Dictionary mapping driver IDs to their decimal odds
        method : str, optional
            Conversion method, by default "basic"
        **kwargs
            Additional parameters for converter

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions

        Examples
        --------
        >>> odds = {"VER": 5, "HAM": 7.75, "NOR": 3}
        >>> dists = DistributionFactory.from_odds_dict(odds)
        """
        # Create simplified structure with only position 1
        structure = {"race": {1: odds_dict}}

        # Use the main method
        distributions = DistributionFactory.from_structured_odds(
            structure, method=method, **kwargs
        )

        # Return just the race distributions
        return distributions.get("race", {})

    @staticmethod
    def from_simple_odds(
        odds: List[float],
        driver_ids: Optional[List[str]] = None,
        method: str = "basic",
        **kwargs,
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions from a list of odds.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds for drivers
        driver_ids : Optional[List[str]], optional
            List of driver IDs, by default None (uses positions as IDs)
        method : str, optional
            Conversion method, by default "basic"
        **kwargs
            Additional parameters for converter

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        if driver_ids is None:
            driver_ids = [str(i + 1) for i in range(len(odds))]

        if len(driver_ids) != len(odds):
            raise ValueError("Number of driver IDs must match number of odds")

        odds_dict = {driver_id: odd for driver_id, odd in zip(driver_ids, odds)}
        return DistributionFactory.from_odds_dict(odds_dict, method=method, **kwargs)

"""
Odds conversion utilities for GridRival F1 fantasy sports.

This module provides conversion tools for transforming betting odds into probability
distributions, which can be used for Monte Carlo simulations and race outcome
predictions. When simulating F1 races for fantasy sports optimization, converting market
odds into accurate probability distributions is a critical first step.

The module is built around the OddsConverter abstract base class, which defines a
common interface for different odds conversion strategies. Concrete implementations
can handle different approaches to margin removal, bookmaker bias, and normalization.

Examples
--------
>>> from gridrival_ai.odds.converters import BasicConverter
>>> # Convert decimal odds to probabilities
>>> converter = BasicConverter()
>>> race_odds = [2.0, 4.0, 10.0, 15.0]  # Decimal odds for top 4 drivers
>>> probabilities = converter.convert(race_odds)
>>> position_dist = converter.to_position_distribution(race_odds)
>>> # Use the distribution in simulations
>>> position_dist.sample(1000)  # Get 1000 random position outcomes

Notes
-----
Betting odds inherently contain a bookmaker's margin, which must be removed
to obtain accurate probabilities. Different converter implementations handle
this margin removal through different mathematical approaches.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from gridrival_ai.probabilities.core import PositionDistribution


class OddsConverter(ABC):
    """
    Abstract base class for odds conversion strategies.

    This class defines the interface for converting betting odds to
    probabilities. Concrete implementations handle different methods
    for margin removal and bias adjustment.
    """

    @abstractmethod
    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.
            Use values > 1.0 for markets like "top N finish".

        Returns
        -------
        np.ndarray
            Array of probabilities summing to target_sum.
        """
        pass

    def convert_to_dict(
        self, odds: List[float], target_sum: float = 1.0
    ) -> Dict[int, float]:
        """
        Convert odds to probability dictionary.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping positions (1-based) to probabilities.
        """
        probs = self.convert(odds, target_sum)
        return {i + 1: float(p) for i, p in enumerate(probs)}

    def to_position_distribution(
        self, odds: List[float], target_sum: float = 1.0, validate: bool = True
    ) -> PositionDistribution:
        """
        Convert odds to position distribution.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.
        validate : bool, optional
            Whether to validate the distribution, by default True.

        Returns
        -------
        PositionDistribution
            Distribution over positions.

        Examples
        --------
        >>> converter = BasicConverter()
        >>> odds = [1.5, 3.0, 6.0]
        >>> dist = converter.to_position_distribution(odds)
        >>> dist[1]  # Probability of P1
        0.5714285714285714
        """
        probs_dict = self.convert_to_dict(odds, target_sum)
        return PositionDistribution(probs_dict, _validate=validate)

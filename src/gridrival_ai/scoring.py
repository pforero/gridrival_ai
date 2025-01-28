"""
Scoring module for GridRival AI.

This module provides the Scorer class for calculating fantasy points.
"""

from typing import Dict, Optional


class Scorer:
    """
    A class to calculate fantasy points for drivers and constructors.

    Parameters
    ----------
    points_system : Dict[str, float], optional
        Custom points system mapping. If None, uses default system.
    """

    def __init__(self, points_system: Optional[Dict[str, float]] = None) -> None:
        """Initialize the scorer with a points system."""
        self.points_system = points_system or {
            "win": 25.0,
            "podium": 15.0,
            "points_finish": 10.0,
            "fastest_lap": 5.0,
        }

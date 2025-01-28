"""
Team management module for GridRival AI.

This module provides the Team class for managing F1 fantasy teams,
including validation of team composition and budget constraints.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class Team:
    """
    A class to represent and manage an F1 fantasy team.

    Parameters
    ----------
    budget : float
        Total budget available for the team.
    drivers : List[str], optional
        List of driver names to initialize the team with.
    constructor : str, optional
        Constructor name to initialize the team with.

    Attributes
    ----------
    MAX_DRIVERS : int
        Maximum number of drivers allowed in a team (2).
    budget : float
        Current available budget.
    drivers : List[str]
        List of current drivers in the team.
    constructor : Optional[str]
        Current constructor in the team.
    total_value : float
        Total value of all contracts in the team.

    Examples
    --------
    >>> team = Team(budget=100.0)
    >>> team.add_driver("Max Verstappen", contract_value=45.0)
    >>> team.add_constructor("Red Bull Racing", contract_value=35.0)
    """

    MAX_DRIVERS = 2

    def __init__(
        self,
        budget: float,
        drivers: Optional[List[str]] = None,
        constructor: Optional[str] = None,
    ) -> None:
        """Initialize a new Team instance."""
        self.budget = budget
        self.drivers = drivers or []
        self.constructor = constructor
        self.total_value = 0.0
        self._contracts: Dict[str, float] = {}

    def add_driver(self, driver: str, contract_value: float) -> bool:
        """
        Add a driver to the team.

        Parameters
        ----------
        driver : str
            Name of the driver to add.
        contract_value : float
            Contract value for the driver.

        Returns
        -------
        bool
            True if driver was successfully added, False otherwise.

        Raises
        ------
        ValueError
            If adding the driver would exceed budget or team size limits.
        """
        if len(self.drivers) >= self.MAX_DRIVERS:
            raise ValueError("Maximum number of drivers already reached")
        
        if contract_value > self.budget:
            raise ValueError("Contract value exceeds available budget")
        
        self.drivers.append(driver)
        self._contracts[driver] = contract_value
        self.budget -= contract_value
        self.total_value += contract_value
        return True

    def add_constructor(self, constructor: str, contract_value: float) -> bool:
        """
        Add a constructor to the team.

        Parameters
        ----------
        constructor : str
            Name of the constructor to add.
        contract_value : float
            Contract value for the constructor.

        Returns
        -------
        bool
            True if constructor was successfully added, False otherwise.

        Raises
        ------
        ValueError
            If adding the constructor would exceed budget or if team already has a constructor.
        """
        if self.constructor is not None:
            raise ValueError("Team already has a constructor")
        
        if contract_value > self.budget:
            raise ValueError("Contract value exceeds available budget")
        
        self.constructor = constructor
        self._contracts[constructor] = contract_value
        self.budget -= contract_value
        self.total_value += contract_value
        return True

    def is_valid(self) -> Tuple[bool, str]:
        """
        Check if the team composition is valid.

        Returns
        -------
        Tuple[bool, str]
            A tuple containing a boolean indicating validity and a message explaining
            any issues found.
        """
        if len(self.drivers) < self.MAX_DRIVERS:
            return False, "Team needs exactly 2 drivers"
        
        if self.constructor is None:
            return False, "Team needs a constructor"
        
        if self.budget < 0:
            return False, "Team exceeds budget"
        
        return True, "Team is valid"

    def __str__(self) -> str:
        """
        Return a string representation of the team.

        Returns
        -------
        str
            A formatted string showing team composition and values.
        """
        team_str = "F1 Fantasy Team:\n"
        team_str += "Drivers:\n"
        for driver in self.drivers:
            team_str += f"  - {driver} (${self._contracts[driver]:.1f}M)\n"
        team_str += f"Constructor: {self.constructor}"
        if self.constructor:
            team_str += f" (${self._contracts[self.constructor]:.1f}M)"
        team_str += f"\nTotal Value: ${self.total_value:.1f}M"
        team_str += f"\nRemaining Budget: ${self.budget:.1f}M"
        return team_str 
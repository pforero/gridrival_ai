"""
Salary management module for GridRival F1 Fantasy League.

This module handles salary adjustments for drivers and constructors
based on their performance and GridRival's constraints.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass
class SalaryConfig:
    """
    Configuration for salary adjustments.

    Parameters
    ----------
    driver_max_change : float
        Maximum allowed change for driver salaries (default: 2.0).
    constructor_max_change : float
        Maximum allowed change for constructor salaries (default: 3.0).
    rounding_increment : float
        Increment to round salaries to (default: 0.1).
    adjustment_factor : float
        Factor to divide salary differences by (default: 4.0).
    """

    driver_max_change: float = 2.0
    constructor_max_change: float = 3.0
    rounding_increment: float = 0.1
    adjustment_factor: float = 4.0


class SalaryManager:
    """
    Manages salary adjustments for drivers and constructors.

    Parameters
    ----------
    config : SalaryConfig, optional
        Configuration for salary adjustments.
    reference_salaries : Dict[int, float], optional
        Reference salaries by rank position.
    """

    def __init__(
        self,
        config: SalaryConfig = SalaryConfig(),
        reference_salaries: Optional[Dict[int, float]] = None,
    ) -> None:
        self.config = config
        self._reference_salaries = (
            reference_salaries or self._default_reference_salaries()
        )

    def _default_reference_salaries(self) -> Dict[int, float]:
        """
        Create default reference salaries based on rank.

        Returns
        -------
        Dict[int, float]
            Mapping of rank to reference salary.
        """
        return {
            1: 35.0,  # Top performer
            2: 32.0,
            3: 30.0,
            4: 28.0,
            5: 26.0,
            6: 24.0,
            7: 22.0,
            8: 20.0,
            9: 18.0,
            10: 16.0,
            11: 14.0,
            12: 12.0,
            13: 10.0,
            14: 9.0,
            15: 8.0,
            16: 7.0,
            17: 6.0,
            18: 5.0,
            19: 4.0,
            20: 3.0,  # Lowest performer
        }

    def _round_to_increment(self, value: float) -> float:
        """
        Round a value to the nearest increment.

        Parameters
        ----------
        value : float
            Value to round.

        Returns
        -------
        float
            Rounded value.
        """
        increment = self.config.rounding_increment
        return round(value / increment) * increment

    def _clamp_change(self, change: float, is_constructor: bool = False) -> float:
        """
        Clamp salary change to allowed limits.

        Parameters
        ----------
        change : float
            Proposed salary change.
        is_constructor : bool, optional
            Whether this is for a constructor, by default False.

        Returns
        -------
        float
            Clamped salary change.
        """
        max_change = (
            self.config.constructor_max_change
            if is_constructor
            else self.config.driver_max_change
        )
        return max(min(change, max_change), -max_change)

    def update_salary(
        self, current_salary: float, rank: int, is_constructor: bool = False
    ) -> float:
        """
        Update salary based on rank performance.

        Parameters
        ----------
        current_salary : float
            Current salary value.
        rank : int
            Performance rank (1-20).
        is_constructor : bool, optional
            Whether this is a constructor salary, by default False.

        Returns
        -------
        float
            New salary value.

        Raises
        ------
        ValueError
            If rank is not between 1 and 20.
        """
        if not 1 <= rank <= 20:
            raise ValueError("Rank must be between 1 and 20")

        reference_salary = self._reference_salaries[rank]
        raw_change = (reference_salary - current_salary) / self.config.adjustment_factor
        clamped_change = self._clamp_change(raw_change, is_constructor)
        new_salary = current_salary + clamped_change
        return self._round_to_increment(new_salary)

    def update_driver_salary(self, current_salary: float, rank: int) -> float:
        """
        Update driver salary based on rank performance.

        Parameters
        ----------
        current_salary : float
            Current salary value.
        rank : int
            Performance rank (1-20).

        Returns
        -------
        float
            New salary value.
        """
        return self.update_salary(current_salary, rank, is_constructor=False)

    def update_constructor_salary(self, current_salary: float, rank: int) -> float:
        """
        Update constructor salary based on rank performance.

        Parameters
        ----------
        current_salary : float
            Current salary value.
        rank : int
            Performance rank (1-20).

        Returns
        -------
        float
            New salary value.
        """
        return self.update_salary(current_salary, rank, is_constructor=True)

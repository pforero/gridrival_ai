"""
Salary management module for GridRival AI.

This module provides the SalaryManager class for handling salary changes.
"""

from typing import Dict, Optional


class SalaryManager:
    """
    A class to manage salary changes for drivers and constructors.

    Parameters
    ----------
    initial_values : Dict[str, float], optional
        Initial salary values for drivers/constructors.
    """

    def __init__(self, initial_values: Optional[Dict[str, float]] = None) -> None:
        """Initialize the salary manager."""
        self.values = initial_values or {} 
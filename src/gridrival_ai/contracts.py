"""
Contract management module for GridRival AI.

This module provides the Contract class for managing F1 fantasy contracts.
"""

from datetime import datetime
from typing import Optional


class Contract:
    """
    A class to manage F1 fantasy contracts.

    Parameters
    ----------
    name : str
        Name of the driver or constructor.
    value : float
        Contract value.
    """

    def __init__(self, name: str, value: float) -> None:
        """Initialize a new contract."""
        self.name = name
        self.value = value 
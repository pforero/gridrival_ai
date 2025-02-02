"""
Contract management module for GridRival F1 Fantasy League.

This module handles contract logic including duration tracking,
early release penalties, and one-race-interval rules.
"""

from dataclasses import dataclass


@dataclass
class Contract:
    """
    Represents a GridRival contract for a driver or constructor.

    Parameters
    ----------
    element_id : int
        Unique identifier for the driver or constructor.
    races_remaining : int
        Number of races left in the contract.
    salary : float
        Current salary value of the contract.
    active : bool, optional
        Whether the contract is currently active, by default True.

    Attributes
    ----------
    element_id : int
        Unique identifier for the driver or constructor.
    races_remaining : int
        Number of races left in the contract.
    salary : float
        Current salary value of the contract.
    active : bool
        Whether the contract is currently active.
    """

    element_id: int
    races_remaining: int
    salary: float
    active: bool = True

    def __post_init__(self) -> None:
        """Validate contract initialization."""
        if self.races_remaining < 0:
            raise ValueError("races_remaining cannot be negative")
        if self.salary < 0:
            raise ValueError("salary cannot be negative")

    def decrement_race(self) -> None:
        """
        Decrease the remaining races by 1 and update active status.

        If races_remaining reaches 0, the contract becomes inactive.
        """
        if not self.active:
            return

        self.races_remaining = max(0, self.races_remaining - 1)
        if self.races_remaining == 0:
            self.active = False


class ContractManager:
    """
    Manages contracts and enforces GridRival rules.

    Parameters
    ----------
    early_release_penalty_rate : float, optional
        Rate for early release penalty, by default 0.03 (3%).

    Attributes
    ----------
    early_release_penalty_rate : float
        Rate applied for early contract termination.
    released_elements : dict
        Tracks when elements were last released (race number).
    current_race : int
        Current race number in the season.
    """

    def __init__(self, early_release_penalty_rate: float = 0.03) -> None:
        self.early_release_penalty_rate = early_release_penalty_rate
        self.released_elements: dict = {}
        self.current_race = 1

    def apply_early_release_penalty(self, contract: Contract) -> float:
        """
        Calculate early release penalty for a contract.

        Parameters
        ----------
        contract : Contract
            Contract to calculate penalty for.

        Returns
        -------
        float
            Penalty amount (3% of contract salary).
        """
        return contract.salary * self.early_release_penalty_rate

    def can_sign_element(self, element_id: int) -> bool:
        """
        Check if an element can be signed based on one-race-interval rule.

        Parameters
        ----------
        element_id : int
            ID of element to check.

        Returns
        -------
        bool
            True if element can be signed, False otherwise.
        """
        last_release = self.released_elements.get(element_id)
        if last_release is None:
            return True
        return self.current_race - last_release > 1

    def release_element(self, element_id: int) -> None:
        """
        Record the release of an element.

        Parameters
        ----------
        element_id : int
            ID of element being released.
        """
        self.released_elements[element_id] = self.current_race

    def advance_race(self) -> None:
        """Advance to the next race."""
        self.current_race += 1

"""
Team management module for GridRival F1 Fantasy League.

This module handles team composition, budget management, and contract integration
for managing a team of 5 drivers and 1 constructor.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from gridrival_ai.contracts import Contract


@dataclass
class Team:
    """
    Represents a GridRival fantasy team with drivers and constructor.

    Parameters
    ----------
    bank_balance : float
        Available budget for the team.
    driver_contracts : List[Contract], optional
        List of current driver contracts, by default empty list.
    constructor_contract : Optional[Contract], optional
        Current constructor contract, by default None.
    max_drivers : int, optional
        Maximum number of drivers allowed, by default 5.

    Attributes
    ----------
    bank_balance : float
        Current available budget.
    driver_contracts : List[Contract]
        List of active driver contracts.
    constructor_contract : Optional[Contract]
        Current constructor contract.
    max_drivers : int
        Maximum number of drivers allowed.
    """

    bank_balance: float
    driver_contracts: List[Contract] = field(default_factory=list)
    constructor_contract: Optional[Contract] = None
    max_drivers: int = 5

    def __post_init__(self) -> None:
        """Validate team initialization."""
        if self.bank_balance < 0:
            raise ValueError("bank_balance cannot be negative")
        if len(self.driver_contracts) > self.max_drivers:
            raise ValueError(f"Cannot have more than {self.max_drivers} drivers")

    def _can_afford(self, contract: Contract) -> bool:
        """
        Check if team can afford a contract.

        Parameters
        ----------
        contract : Contract
            Contract to check affordability for.

        Returns
        -------
        bool
            True if contract is affordable, False otherwise.
        """
        return self.bank_balance >= contract.salary

    def add_driver(self, contract: Contract) -> None:
        """
        Add a driver contract to the team.

        Parameters
        ----------
        contract : Contract
            Driver contract to add.

        Raises
        ------
        ValueError
            If team is full or cannot afford the contract.
        """
        if len(self.driver_contracts) >= self.max_drivers:
            raise ValueError(f"Team already has {self.max_drivers} drivers")
        if not self._can_afford(contract):
            raise ValueError("Insufficient funds for this contract")

        self.bank_balance -= contract.salary
        self.driver_contracts.append(contract)

    def add_constructor(self, contract: Contract) -> None:
        """
        Add a constructor contract to the team.

        Parameters
        ----------
        contract : Contract
            Constructor contract to add.

        Raises
        ------
        ValueError
            If team already has a constructor or cannot afford the contract.
        """
        if self.constructor_contract is not None:
            raise ValueError("Team already has a constructor")
        if not self._can_afford(contract):
            raise ValueError("Insufficient funds for this contract")

        self.bank_balance -= contract.salary
        self.constructor_contract = contract

    def remove_driver(self, element_id: str) -> None:
        """
        Remove a driver contract from the team.

        Parameters
        ----------
        element_id : str
            Three-letter abbreviation of the driver to remove.

        Raises
        ------
        ValueError
            If driver is not found in the team.
        """
        for contract in self.driver_contracts:
            if contract.element_id == element_id:
                self.bank_balance += contract.salary
                self.driver_contracts.remove(contract)
                return
        raise ValueError(f"Driver with ID {element_id} not found in team")

    def remove_constructor(self) -> None:
        """
        Remove the constructor contract from the team.

        Raises
        ------
        ValueError
            If team has no constructor.
        """
        if self.constructor_contract is None:
            raise ValueError("Team has no constructor to remove")

        self.bank_balance += self.constructor_contract.salary
        self.constructor_contract = None

    def update_after_race(self) -> None:
        """
        Update team after a race.

        Decrements race counts and removes expired contracts.
        """
        # Update drivers
        active_drivers = []
        for contract in self.driver_contracts:
            contract.decrement_race()
            if contract.active:
                active_drivers.append(contract)
            else:
                self.bank_balance += contract.salary
        self.driver_contracts = active_drivers

        # Update constructor
        if self.constructor_contract is not None:
            self.constructor_contract.decrement_race()
            if not self.constructor_contract.active:
                self.bank_balance += self.constructor_contract.salary
                self.constructor_contract = None

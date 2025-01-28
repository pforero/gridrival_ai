"""Tests for the team module."""

import pytest

from gridrival_ai.contracts import Contract
from gridrival_ai.team import Team


@pytest.fixture
def empty_team():
    """Create an empty team with 100M budget."""
    return Team(bank_balance=100.0)


@pytest.fixture
def driver_contract():
    """Create a sample driver contract."""
    return Contract(element_id=1, races_remaining=5, salary=20.0)


@pytest.fixture
def constructor_contract():
    """Create a sample constructor contract."""
    return Contract(element_id=101, races_remaining=5, salary=30.0)


def test_team_initialization():
    """Test team creation with valid and invalid parameters."""
    # Valid team
    team = Team(bank_balance=100.0)
    assert team.bank_balance == 100.0
    assert len(team.driver_contracts) == 0
    assert team.constructor_contract is None

    # Invalid team
    with pytest.raises(ValueError):
        Team(bank_balance=-100.0)


def test_add_driver(empty_team, driver_contract):
    """Test adding drivers to team."""
    # Add first driver
    empty_team.add_driver(driver_contract)
    assert len(empty_team.driver_contracts) == 1
    assert empty_team.bank_balance == 80.0  # 100 - 20

    # Add more drivers until full
    for i in range(2, 6):
        contract = Contract(element_id=i, races_remaining=5, salary=10.0)
        empty_team.add_driver(contract)

    # Try to add one more
    with pytest.raises(ValueError):
        contract = Contract(element_id=6, races_remaining=5, salary=10.0)
        empty_team.add_driver(contract)


def test_add_constructor(empty_team, constructor_contract):
    """Test adding constructor to team."""
    # Add constructor
    empty_team.add_constructor(constructor_contract)
    assert empty_team.constructor_contract is not None
    assert empty_team.bank_balance == 70.0  # 100 - 30

    # Try to add second constructor
    with pytest.raises(ValueError):
        empty_team.add_constructor(constructor_contract)


def test_insufficient_funds(empty_team):
    """Test handling of expensive contracts."""
    expensive_contract = Contract(element_id=1, races_remaining=5, salary=150.0)
    with pytest.raises(ValueError):
        empty_team.add_driver(expensive_contract)


def test_remove_driver(empty_team, driver_contract):
    """Test removing a driver from team."""
    empty_team.add_driver(driver_contract)
    initial_balance = empty_team.bank_balance

    empty_team.remove_driver(driver_contract.element_id)
    assert len(empty_team.driver_contracts) == 0
    assert empty_team.bank_balance == initial_balance + driver_contract.salary

    # Try to remove non-existent driver
    with pytest.raises(ValueError):
        empty_team.remove_driver(999)


def test_remove_constructor(empty_team, constructor_contract):
    """Test removing constructor from team."""
    empty_team.add_constructor(constructor_contract)
    initial_balance = empty_team.bank_balance

    empty_team.remove_constructor()
    assert empty_team.constructor_contract is None
    assert empty_team.bank_balance == initial_balance + constructor_contract.salary

    # Try to remove when no constructor
    with pytest.raises(ValueError):
        empty_team.remove_constructor()


def test_update_after_race(empty_team):
    """Test team update after race."""
    # Add contracts with different durations
    empty_team.add_driver(Contract(element_id=1, races_remaining=1, salary=10.0))
    empty_team.add_driver(Contract(element_id=2, races_remaining=2, salary=10.0))
    empty_team.add_constructor(Contract(element_id=101, races_remaining=1, salary=20.0))

    initial_balance = empty_team.bank_balance

    # Simulate race
    empty_team.update_after_race()

    # Check expired contracts
    assert len(empty_team.driver_contracts) == 1  # One driver should remain
    assert empty_team.constructor_contract is None  # Constructor should be removed
    assert empty_team.bank_balance == initial_balance + 30.0  # Refunded 10 + 20

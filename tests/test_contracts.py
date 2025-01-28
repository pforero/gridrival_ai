"""Tests for the contracts module."""

import pytest

from gridrival_ai.contracts import Contract, ContractManager


def test_contract_initialization():
    """Test contract creation with valid and invalid parameters."""
    # Valid contract
    contract = Contract(element_id=1, races_remaining=5, salary=10.0)
    assert contract.element_id == 1
    assert contract.races_remaining == 5
    assert contract.salary == 10.0
    assert contract.active is True

    # Invalid contracts
    with pytest.raises(ValueError):
        Contract(element_id=1, races_remaining=-1, salary=10.0)
    with pytest.raises(ValueError):
        Contract(element_id=1, races_remaining=5, salary=-10.0)


def test_contract_decrement_race():
    """Test race decrement behavior."""
    contract = Contract(element_id=1, races_remaining=2, salary=10.0)

    # First decrement
    contract.decrement_race()
    assert contract.races_remaining == 1
    assert contract.active is True

    # Second decrement
    contract.decrement_race()
    assert contract.races_remaining == 0
    assert contract.active is False

    # Further decrements should have no effect
    contract.decrement_race()
    assert contract.races_remaining == 0
    assert contract.active is False


def test_early_release_penalty():
    """Test early release penalty calculation."""
    manager = ContractManager()
    contract = Contract(element_id=1, races_remaining=5, salary=100.0)

    penalty = manager.apply_early_release_penalty(contract)
    assert penalty == pytest.approx(3.0)  # 3% of 100.0


def test_one_race_interval_rule():
    """Test enforcement of one-race-interval rule."""
    manager = ContractManager()

    # Initially can sign any element
    assert manager.can_sign_element(1) is True

    # Release element 1
    manager.release_element(1)
    assert manager.can_sign_element(1) is False

    # Advance one race
    manager.advance_race()
    assert manager.can_sign_element(1) is False

    # Advance another race
    manager.advance_race()
    assert manager.can_sign_element(1) is True


def test_multiple_elements_tracking():
    """Test tracking of multiple elements."""
    manager = ContractManager()

    # Release two elements
    manager.release_element(1)
    manager.release_element(2)

    assert manager.can_sign_element(1) is False
    assert manager.can_sign_element(2) is False

    # Advance races
    manager.advance_race()
    manager.advance_race()

    assert manager.can_sign_element(1) is True
    assert manager.can_sign_element(2) is True


def test_contract_manager_race_progression():
    """Test race progression in contract manager."""
    manager = ContractManager()
    assert manager.current_race == 1

    manager.advance_race()
    assert manager.current_race == 2

    manager.advance_race()
    assert manager.current_race == 3

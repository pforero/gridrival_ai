"""Tests for the salary module."""

import pytest

from gridrival_ai.salary import SalaryConfig, SalaryManager


@pytest.fixture
def salary_manager():
    """Create a SalaryManager instance for testing."""
    return SalaryManager()


def test_driver_salary_update_basic(salary_manager):
    """Test basic driver salary updates."""
    # Test improvement (rank 1)
    new_salary = salary_manager.update_driver_salary(current_salary=20.0, rank=1)
    assert new_salary > 20.0
    assert new_salary <= 22.0  # Max increase of 2.0

    # Test decline (rank 20)
    new_salary = salary_manager.update_driver_salary(current_salary=20.0, rank=20)
    assert new_salary < 20.0
    assert new_salary >= 18.0  # Max decrease of 2.0


def test_constructor_salary_update_basic(salary_manager):
    """Test basic constructor salary updates."""
    # Test improvement (rank 1)
    new_salary = salary_manager.update_constructor_salary(current_salary=30.0, rank=1)
    assert new_salary > 30.0
    assert new_salary <= 33.0  # Max increase of 3.0

    # Test decline (rank 10)
    new_salary = salary_manager.update_constructor_salary(current_salary=30.0, rank=10)
    assert new_salary < 30.0
    assert new_salary >= 27.0  # Max decrease of 3.0


def test_salary_rounding():
    """Test salary rounding to 0.1M increments."""
    manager = SalaryManager(config=SalaryConfig(rounding_increment=0.1))

    # Test various scenarios
    new_salary = manager.update_driver_salary(current_salary=20.0, rank=1)
    decimal_places = len(str(new_salary).split(".")[-1])
    assert decimal_places <= 1

    # Check that the value is actually rounded to 0.1
    assert (new_salary * 10) % 1 == 0


def test_invalid_rank(salary_manager):
    """Test handling of invalid rank values."""
    with pytest.raises(ValueError):
        salary_manager.update_driver_salary(current_salary=20.0, rank=0)

    with pytest.raises(ValueError):
        salary_manager.update_driver_salary(current_salary=20.0, rank=21)


def test_extreme_salary_changes():
    """Test handling of extreme salary differences."""
    manager = SalaryManager()

    # Test large positive change for driver
    new_salary = manager.update_driver_salary(current_salary=5.0, rank=1)
    assert new_salary <= 7.0  # Max increase of 2.0

    # Test large negative change for constructor
    new_salary = manager.update_constructor_salary(current_salary=35.0, rank=20)
    assert new_salary >= 32.0  # Max decrease of 3.0


def test_custom_config():
    """Test SalaryManager with custom configuration."""
    config = SalaryConfig(
        driver_max_change=1.0,
        constructor_max_change=2.0,
        rounding_increment=0.5,
        adjustment_factor=2.0,
    )
    manager = SalaryManager(config=config)

    # Test driver with custom limits
    new_salary = manager.update_driver_salary(current_salary=20.0, rank=1)
    assert new_salary <= 21.0  # Max increase of 1.0
    assert (new_salary * 2) % 1 == 0  # Check 0.5 rounding

    # Test constructor with custom limits
    new_salary = manager.update_constructor_salary(current_salary=30.0, rank=20)
    assert new_salary >= 28.0  # Max decrease of 2.0
    assert (new_salary * 2) % 1 == 0  # Check 0.5 rounding


def test_reference_salaries():
    """Test custom reference salaries."""
    custom_references = {
        1: 40.0,
        2: 35.0,
        # ... only testing top positions
        19: 5.0,
        20: 4.0,
    }
    manager = SalaryManager(reference_salaries=custom_references)

    # Test with custom reference values
    new_salary = manager.update_driver_salary(current_salary=20.0, rank=1)
    assert new_salary > 20.0

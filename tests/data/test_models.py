"""Tests for core F1 data models."""

import pytest

from gridrival_ai.data.models import Constructor, Driver, Race


def test_driver_creation():
    """Test basic driver creation and validation."""
    # Valid driver
    driver = Driver(driver_id="VER", name="Max Verstappen")
    assert driver.driver_id == "VER"
    assert driver.name == "Max Verstappen"

    # Invalid driver_id length
    with pytest.raises(ValueError, match="must be exactly 3 letters"):
        Driver(driver_id="VERST", name="Max Verstappen")

    # Invalid driver_id characters
    with pytest.raises(ValueError, match="must contain only letters"):
        Driver(driver_id="V3R", name="Max Verstappen")


def test_constructor_creation():
    """Test basic constructor creation and validation."""
    # Valid constructor
    constructor = Constructor(
        constructor_id="RBR",
        name="Red Bull Racing",
        drivers=("VER", "PER"),
    )
    assert constructor.constructor_id == "RBR"
    assert constructor.name == "Red Bull Racing"
    assert len(constructor.drivers) == 2
    assert constructor.drivers[0] == "VER"
    assert constructor.drivers[1] == "PER"

    # Invalid number of drivers
    with pytest.raises(ValueError, match="must have exactly 2 drivers"):
        Constructor(
            constructor_id="RBR",
            name="Red Bull Racing",
            drivers=("VER",),
        )

    # Invalid constructor_id length
    with pytest.raises(ValueError, match="must be exactly 3 letters"):
        Constructor(
            constructor_id="REDB",
            name="Red Bull Racing",
            drivers=("VER", "PER"),
        )

    # Invalid constructor_id characters
    with pytest.raises(ValueError, match="must contain only letters"):
        Constructor(
            constructor_id="RB1",
            name="Red Bull Racing",
            drivers=("VER", "PER"),
        )

    # Invalid driver_id
    with pytest.raises(ValueError, match="Driver ID must be exactly 3 letters"):
        Constructor(
            constructor_id="RBR",
            name="Red Bull Racing",
            drivers=("VER", "PEREZ"),
        )


def test_race_creation():
    """Test basic race creation and validation."""
    # Valid races
    race1 = Race(name="Monaco", is_sprint=False)
    assert race1.name == "Monaco"
    assert not race1.is_sprint

    race2 = Race(name="Brazil", is_sprint=True)
    assert race2.name == "Brazil"
    assert race2.is_sprint

    # Empty name
    with pytest.raises(ValueError, match="Race name cannot be empty"):
        Race(name="", is_sprint=False)

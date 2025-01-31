"""Tests for core F1 data structures."""

import pytest

from gridrival_ai.f1.core import Constructor, Pilot, Race


def test_pilot_creation():
    """Test basic pilot creation and validation."""
    # Valid pilot
    pilot = Pilot(driver_id="VER", name="Max Verstappen", abbreviation="VER")
    assert pilot.driver_id == "VER"
    assert pilot.name == "Max Verstappen"
    assert pilot.abbreviation == "VER"

    # Invalid abbreviation length
    with pytest.raises(ValueError, match="must be exactly 3 letters"):
        Pilot(driver_id="VER", name="Max Verstappen", abbreviation="VERST")

    # Invalid abbreviation characters
    with pytest.raises(ValueError, match="must contain only letters"):
        Pilot(driver_id="VER", name="Max Verstappen", abbreviation="V3R")


def test_constructor_creation():
    """Test basic constructor creation and validation."""
    pilot1 = Pilot(driver_id="VER", name="Max Verstappen", abbreviation="VER")
    pilot2 = Pilot(driver_id="PER", name="Sergio Perez", abbreviation="PER")

    # Valid constructor
    constructor = Constructor(
        constructor_id="RBR",
        name="Red Bull Racing",
        abbreviation="RBR",
        drivers=[pilot1, pilot2],
    )
    assert constructor.constructor_id == "RBR"
    assert constructor.name == "Red Bull Racing"
    assert constructor.abbreviation == "RBR"
    assert len(constructor.drivers) == 2
    assert constructor.drivers[0] == pilot1
    assert constructor.drivers[1] == pilot2

    # Invalid number of drivers
    with pytest.raises(ValueError, match="must have exactly 2 drivers"):
        Constructor(
            constructor_id="RBR",
            name="Red Bull Racing",
            abbreviation="RBR",
            drivers=[pilot1],
        )

    # Invalid abbreviation length
    with pytest.raises(ValueError, match="must be exactly 3 letters"):
        Constructor(
            constructor_id="RBR",
            name="Red Bull Racing",
            abbreviation="REDB",
            drivers=[pilot1, pilot2],
        )

    # Invalid abbreviation characters
    with pytest.raises(ValueError, match="must contain only letters"):
        Constructor(
            constructor_id="RBR",
            name="Red Bull Racing",
            abbreviation="RB1",
            drivers=[pilot1, pilot2],
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

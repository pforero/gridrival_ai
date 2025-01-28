"""Tests for the data ingestion module."""

import io
from pathlib import Path

import pandas as pd
import pytest

from gridrival_ai.data_ingestion import load_constructor_data, load_driver_data


@pytest.fixture
def valid_driver_csv():
    """Create a valid driver data CSV in memory."""
    return io.StringIO(
        """driver_id,name,salary,rolling_avg_finish
VER,Max Verstappen,45.5,2.1
HAM,Lewis Hamilton,42.0,3.5
LEC,Charles Leclerc,38.5,4.2"""
    )


@pytest.fixture
def valid_constructor_csv():
    """Create a valid constructor data CSV in memory."""
    return io.StringIO(
        """constructor_id,name,salary,rolling_avg_points
RBR,Red Bull Racing,35.5,45.5
MER,Mercedes,32.0,40.0
FER,Ferrari,30.5,38.5"""
    )


@pytest.fixture
def minimal_driver_csv():
    """Create a driver data CSV with only required columns."""
    return io.StringIO(
        """driver_id,name,salary
VER,Max Verstappen,45.5
HAM,Lewis Hamilton,42.0
LEC,Charles Leclerc,38.5"""
    )


@pytest.fixture
def minimal_constructor_csv():
    """Create a constructor data CSV with only required columns."""
    return io.StringIO(
        """constructor_id,name,salary
RBR,Red Bull Racing,35.5
MER,Mercedes,32.0
FER,Ferrari,30.5"""
    )


def test_load_driver_data_full(tmp_path, valid_driver_csv):
    """Test loading driver data with all columns present."""
    # Save the CSV data to a temporary file
    csv_path = tmp_path / "drivers.csv"
    csv_path.write_text(valid_driver_csv.getvalue())

    # Load the data
    df = load_driver_data(str(csv_path))

    # Check the DataFrame properties
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["driver_id", "name", "salary", "rolling_avg_finish"]
    assert df["driver_id"].iloc[0] == "VER"
    assert df["name"].iloc[0] == "Max Verstappen"
    assert df["salary"].iloc[0] == 45.5
    assert df["rolling_avg_finish"].iloc[0] == 2.1


def test_load_driver_data_minimal(tmp_path, minimal_driver_csv):
    """Test loading driver data with only required columns."""
    # Save the CSV data to a temporary file
    csv_path = tmp_path / "drivers_minimal.csv"
    csv_path.write_text(minimal_driver_csv.getvalue())

    # Load the data
    df = load_driver_data(str(csv_path))

    # Check that optional columns were added with default values
    assert "rolling_avg_finish" in df.columns
    assert pd.isna(df["rolling_avg_finish"].iloc[0])


def test_load_constructor_data_full(tmp_path, valid_constructor_csv):
    """Test loading constructor data with all columns present."""
    # Save the CSV data to a temporary file
    csv_path = tmp_path / "constructors.csv"
    csv_path.write_text(valid_constructor_csv.getvalue())

    # Load the data
    df = load_constructor_data(str(csv_path))

    # Check the DataFrame properties
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["constructor_id", "name", "salary", "rolling_avg_points"]
    assert df["constructor_id"].iloc[0] == "RBR"
    assert df["name"].iloc[0] == "Red Bull Racing"
    assert df["salary"].iloc[0] == 35.5
    assert df["rolling_avg_points"].iloc[0] == 45.5


def test_load_constructor_data_minimal(tmp_path, minimal_constructor_csv):
    """Test loading constructor data with only required columns."""
    # Save the CSV data to a temporary file
    csv_path = tmp_path / "constructors_minimal.csv"
    csv_path.write_text(minimal_constructor_csv.getvalue())

    # Load the data
    df = load_constructor_data(str(csv_path))

    # Check that optional columns were added with default values
    assert "rolling_avg_points" in df.columns
    assert pd.isna(df["rolling_avg_points"].iloc[0])


def test_missing_required_columns_driver(tmp_path):
    """Test error handling when required columns are missing in driver data."""
    # Create CSV with missing required column
    csv_data = """driver_id,name
VER,Max Verstappen
HAM,Lewis Hamilton"""
    
    csv_path = tmp_path / "invalid_drivers.csv"
    csv_path.write_text(csv_data)

    with pytest.raises(ValueError, match="Missing required columns in driver data: salary"):
        load_driver_data(str(csv_path))


def test_missing_required_columns_constructor(tmp_path):
    """Test error handling when required columns are missing in constructor data."""
    # Create CSV with missing required column
    csv_data = """constructor_id,name
RBR,Red Bull Racing
MER,Mercedes"""
    
    csv_path = tmp_path / "invalid_constructors.csv"
    csv_path.write_text(csv_data)

    with pytest.raises(ValueError, match="Missing required columns in constructor data: salary"):
        load_constructor_data(str(csv_path))


def test_invalid_numeric_values_driver(tmp_path):
    """Test error handling when numeric values are invalid in driver data."""
    csv_data = """driver_id,name,salary
VER,Max Verstappen,invalid
HAM,Lewis Hamilton,42.0"""
    
    csv_path = tmp_path / "invalid_numeric_drivers.csv"
    csv_path.write_text(csv_data)

    with pytest.raises(ValueError, match="Invalid numeric values in driver data"):
        load_driver_data(str(csv_path))


def test_invalid_numeric_values_constructor(tmp_path):
    """Test error handling when numeric values are invalid in constructor data."""
    csv_data = """constructor_id,name,salary
RBR,Red Bull Racing,invalid
MER,Mercedes,32.0"""
    
    csv_path = tmp_path / "invalid_numeric_constructors.csv"
    csv_path.write_text(csv_data)

    with pytest.raises(ValueError, match="Invalid numeric values in constructor data"):
        load_constructor_data(str(csv_path))


def test_nonexistent_file():
    """Test error handling when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_driver_data("nonexistent_file.csv") 
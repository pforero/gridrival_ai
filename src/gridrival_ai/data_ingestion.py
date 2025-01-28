"""
Data ingestion module for GridRival AI.

This module provides functions for loading and validating driver and constructor
data from CSV files. It ensures data consistency and provides default values
for optional fields.
"""

from typing import List, Set

import pandas as pd


def _validate_columns(df: pd.DataFrame, required_cols: Set[str], source: str) -> None:
    """
    Validate that all required columns are present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required_cols : Set[str]
        Set of required column names.
    source : str
        Source description for error messages (e.g., 'driver' or 'constructor').

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {source} data: {', '.join(sorted(missing_cols))}"
        )


def load_driver_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate driver data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing driver data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing standardized driver data with columns:
        - driver_id (str): Unique identifier for each driver
        - name (str): Human-readable driver name
        - salary (float): Contract value in millions (e.g., 32.0 for £32M)
        - rolling_avg_finish (float): Average finish position over last 8 races
                                    (optional, defaults to None)

    Raises
    ------
    ValueError
        If required columns are missing or data types are invalid.
    FileNotFoundError
        If the specified file does not exist.
    """
    # Define column order explicitly
    required_cols = ["driver_id", "name", "salary"]
    optional_cols = {
        "rolling_avg_finish": None,
    }

    # Read CSV file
    df = pd.read_csv(filepath)

    # Validate required columns
    _validate_columns(df, set(required_cols), "driver")

    # Add optional columns with default values if missing
    for col, default_value in optional_cols.items():
        if col not in df.columns:
            df[col] = default_value

    # Ensure proper data types
    try:
        df["salary"] = pd.to_numeric(df["salary"])
        if "rolling_avg_finish" in df.columns and df["rolling_avg_finish"].notna().any():
            df["rolling_avg_finish"] = pd.to_numeric(df["rolling_avg_finish"])
    except ValueError as e:
        raise ValueError("Invalid numeric values in driver data") from e

    # Ensure string columns are properly formatted
    df["driver_id"] = df["driver_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    # Return DataFrame with standardized column order
    final_columns = required_cols + list(optional_cols.keys())
    return df[final_columns]


def load_constructor_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate constructor (team) data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing constructor data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing standardized constructor data with columns:
        - constructor_id (str): Unique identifier for each constructor
        - name (str): Human-readable constructor name
        - salary (float): Contract value in millions
        - rolling_avg_points (float): Average points over last 8 races
                                    (optional, defaults to None)

    Raises
    ------
    ValueError
        If required columns are missing or data types are invalid.
    FileNotFoundError
        If the specified file does not exist.
    """
    # Define column order explicitly
    required_cols = ["constructor_id", "name", "salary"]
    optional_cols = {
        "rolling_avg_points": None,
    }

    # Read CSV file
    df = pd.read_csv(filepath)

    # Validate required columns
    _validate_columns(df, set(required_cols), "constructor")

    # Add optional columns with default values if missing
    for col, default_value in optional_cols.items():
        if col not in df.columns:
            df[col] = default_value

    # Ensure proper data types
    try:
        df["salary"] = pd.to_numeric(df["salary"])
        if "rolling_avg_points" in df.columns and df["rolling_avg_points"].notna().any():
            df["rolling_avg_points"] = pd.to_numeric(df["rolling_avg_points"])
    except ValueError as e:
        raise ValueError("Invalid numeric values in constructor data") from e

    # Ensure string columns are properly formatted
    df["constructor_id"] = df["constructor_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    # Return DataFrame with standardized column order
    final_columns = required_cols + list(optional_cols.keys())
    return df[final_columns] 
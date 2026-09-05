"""
Reproducible Data Ingestion Module for Retail Sales Forecasting & Analytics.

Provides clean, reusable functions to load the raw Superstore retail dataset
without business transformations or record dropping.
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd

from src.config import RAW_DATA_FILE

DEFAULT_ENCODING: str = "windows-1252"
DEFAULT_DATE_COLUMNS: List[str] = ["Order Date", "Ship Date"]


def load_raw_data(
    file_path: Optional[Path] = None,
    encoding: str = DEFAULT_ENCODING,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load the raw Superstore retail transaction dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : Optional[Path], default=None
        Path to the raw CSV file. If None, resolves to `data/raw/Sample - Superstore.csv`
        configured in `src.config.RAW_DATA_FILE`.
    encoding : str, default="windows-1252"
        Character encoding of the CSV file.
    parse_dates : bool, default=True
        Whether to parse 'Order Date' and 'Ship Date' columns into datetime objects.
        Dates are parsed in month-first order (standard US MM-DD-YYYY or M/D/YYYY).

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame containing the raw transaction records.

    Raises
    -------
    FileNotFoundError
        If the raw dataset file does not exist at the resolved path.
    ValueError
        If the file cannot be read or parsed with the specified encoding.
    """
    target_path = Path(file_path) if file_path is not None else RAW_DATA_FILE

    if not target_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: '{target_path}'. "
            f"Please ensure the raw data file exists before ingestion."
        )

    try:
        df = pd.read_csv(target_path, encoding=encoding)
    except UnicodeDecodeError as err:
        raise ValueError(
            f"Failed to decode '{target_path}' using encoding '{encoding}'. "
            f"Error: {err}"
        ) from err

    if parse_dates:
        for col in DEFAULT_DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)

    return df

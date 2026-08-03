"""Dataset ingestion + metadata tool (Section 5; Section 6 tools #1/#2).

`load_csv` handles safe ingestion of an uploaded file into a pandas
dataframe. `get_dataset_metadata` wraps the pure-pandas profiling logic
in `app.analytics.profiling` into the `DatasetMetadata` pydantic schema
that is sent to the LLM and returned by the upload API endpoint.
"""

from __future__ import annotations

import io

import pandas as pd

from app.analytics.profiling import profile_dataset
from app.analytics.validation import validate_dataframe
from app.core.exceptions import InvalidCSVError
from app.core.logging import get_logger
from app.models.schemas import ColumnProfile, DatasetMetadata

logger = get_logger(__name__)


def load_csv(file_bytes: bytes, max_rows: int) -> pd.DataFrame:
    """Parse raw uploaded bytes into a validated pandas DataFrame.

    Raises:
        InvalidCSVError: if the bytes cannot be parsed as a CSV.
        EmptyDatasetError / DatasetTooLargeError: via `validate_dataframe`.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InvalidCSVError(f"Could not parse the uploaded file as a CSV: {exc}") from exc

    validate_dataframe(df, max_rows=max_rows)
    logger.info("Loaded CSV with shape %s", df.shape)
    return df


def get_dataset_metadata(df: pd.DataFrame, dataset_id: str, filename: str) -> DatasetMetadata:
    """Build the full `DatasetMetadata` schema for a dataframe."""
    raw = profile_dataset(df)
    columns = [ColumnProfile(**col) for col in raw["columns"]]

    return DatasetMetadata(
        dataset_id=dataset_id,
        filename=filename,
        n_rows=raw["n_rows"],
        n_columns=raw["n_columns"],
        columns=columns,
        duplicate_row_count=raw["duplicate_row_count"],
        numeric_columns=raw["numeric_columns"],
        categorical_columns=raw["categorical_columns"],
        datetime_columns=raw["datetime_columns"],
        boolean_columns=raw["boolean_columns"],
        possible_target_columns=raw["possible_target_columns"],
    )

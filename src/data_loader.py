"""Dataset management: sample data, CSV/TXT upload parsing, cleaning and statistics."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd

from .utils import SAMPLE_CSV

MAX_UPLOAD_MB = 50


class DatasetError(ValueError):
    """Raised for invalid or unusable dataset input."""


@dataclass
class CleanInfo:
    """Summary of the cleaning step applied to a raw dataset."""
    original_rows: int
    missing_removed: int
    duplicates_removed: int
    final_rows: int


def load_sample() -> pd.DataFrame:
    """Load the built-in sample collection (54 documents, 6 themes)."""
    if not SAMPLE_CSV.exists():
        raise DatasetError(f"Sample dataset not found at {SAMPLE_CSV}.")
    return pd.read_csv(SAMPLE_CSV)


def _read_bytes(file: BinaryIO | bytes) -> bytes:
    data = file if isinstance(file, (bytes, bytearray)) else file.read()
    if not data:
        raise DatasetError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DatasetError(f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    return bytes(data)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DatasetError("Could not decode file; please save it as UTF-8.")


def load_csv(file: BinaryIO | bytes) -> pd.DataFrame:
    """Parse an uploaded CSV file into a DataFrame."""
    text = _decode(_read_bytes(file))
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # pandas raises several parser error types
        raise DatasetError(f"Could not parse CSV: {exc}") from exc
    if df.empty or df.shape[1] == 0:
        raise DatasetError("The CSV contains no rows or columns.")
    return df


def load_txt(file: BinaryIO | bytes) -> pd.DataFrame:
    """Parse a TXT file where documents are separated by blank lines."""
    text = _decode(_read_bytes(file)).replace("\r\n", "\n")
    docs = [" ".join(block.split()) for block in text.split("\n\n")]
    docs = [d for d in docs if d]
    if not docs:
        raise DatasetError("No documents found. Separate documents with blank lines.")
    return pd.DataFrame({"doc_id": range(1, len(docs) + 1), "text": docs})


def guess_text_column(df: pd.DataFrame) -> str:
    """Pick the object column with the longest average text as the default text column."""
    candidates = [c for c in df.columns
                  if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not candidates:
        raise DatasetError("No text (string) column found in the dataset.")
    return max(candidates, key=lambda c: df[c].astype(str).str.len().mean())


def clean_dataset(df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, CleanInfo]:
    """Drop missing/blank and duplicate documents; returns df with a stable `doc_id`."""
    if text_column not in df.columns:
        raise DatasetError(f"Column '{text_column}' not found.")
    work = df.copy()
    work[text_column] = work[text_column].astype("string")
    n0 = len(work)
    blank = work[text_column].isna() | (work[text_column].str.strip() == "")
    work = work[~blank]
    n_missing = n0 - len(work)
    n1 = len(work)
    work = work.drop_duplicates(subset=[text_column])
    n_dup = n1 - len(work)
    work = work.reset_index(drop=True)
    work[text_column] = work[text_column].astype(str)
    if "doc_id" in work.columns:  # preserve the user's own ids under another name
        work = work.rename(columns={"doc_id": "original_doc_id"})
    work.insert(0, "doc_id", range(len(work)))
    if work.empty:
        raise DatasetError("No usable documents remain after removing missing values.")
    return work, CleanInfo(n0, n_missing, n_dup, len(work))


def dataset_stats(df: pd.DataFrame, text_column: str) -> dict[str, float]:
    """Basic descriptive statistics of document length."""
    lengths = df[text_column].astype(str).str.split().str.len()
    return {"documents": int(len(df)), "mean_words": float(lengths.mean()),
            "median_words": float(lengths.median()), "min_words": int(lengths.min()),
            "max_words": int(lengths.max()), "total_words": int(lengths.sum())}

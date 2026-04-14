from __future__ import annotations

from typing import List
import pandas as pd

from core.config import MAX_UNIQUE_RATIO_FOR_CATEGORICAL
from core.schemas import DatasetProfile
from utils.helpers import DATE_HINTS

# A column is "text-heavy" (free text, bad for charts) when its average
# string length exceeds this threshold.  "Fever, cough, sore throat" ≈ 30 chars
# but "Patient shows signs of seasonal flu." ≈ 55 chars.
_TEXT_AVG_LEN_THRESHOLD = 45


class DataProfiler:
    @staticmethod
    def _detect_datetime_columns(df: pd.DataFrame) -> List[str]:
        datetime_cols: List[str] = []
        for col in df.columns:
            # Skip columns that are already numeric — prevents misclassifying
            # numbers like 7800 (monthly_salary) as timestamps.
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            col_lower = col.lower()
            if any(hint in col_lower for hint in DATE_HINTS):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.6:
                    df[col] = parsed
                    datetime_cols.append(col)
                    continue

            if df[col].dtype == object:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
                    datetime_cols.append(col)

        return datetime_cols

    @staticmethod
    def _detect_id_like_columns(df: pd.DataFrame) -> List[str]:
        """
        Flag columns where almost every value is unique — likely an ID/key.
        Also catches pattern-IDs like "PT-1001", "ORD-0042", UUIDs, etc.
        """
        id_like = []
        total_rows = max(len(df), 1)
        for col in df.columns:
            unique_ratio = df[col].nunique(dropna=True) / total_rows
            if unique_ratio > 0.95 and df[col].dtype == object:
                id_like.append(col)
        return id_like

    @staticmethod
    def _compute_avg_string_lengths(df: pd.DataFrame, columns: List[str]) -> dict:
        result = {}
        for col in columns:
            try:
                avg_len = df[col].dropna().astype(str).str.len().mean()
                result[col] = float(avg_len) if pd.notna(avg_len) else 0.0
            except Exception:
                result[col] = 0.0
        return result

    @staticmethod
    def profile(df: pd.DataFrame) -> DatasetProfile:
        if df.empty or len(df.columns) == 0:
            return DatasetProfile(
                row_count=len(df),
                column_count=len(df.columns),
            )

        working_df = df.copy()
        datetime_columns = DataProfiler._detect_datetime_columns(working_df)
        numeric_columns = working_df.select_dtypes(include=["number"]).columns.tolist()

        # Coerce object columns that are actually numeric (e.g. "440", "580" stored as strings)
        for col in working_df.columns:
            if col not in numeric_columns and col not in datetime_columns and working_df[col].dtype == object:
                coerced = pd.to_numeric(working_df[col], errors="coerce")
                if coerced.notna().mean() > 0.8:
                    working_df[col] = coerced
                    numeric_columns.append(col)

        id_like_columns = DataProfiler._detect_id_like_columns(working_df)

        categorical_columns = []
        for col in working_df.columns:
            if col in numeric_columns or col in datetime_columns or col in id_like_columns:
                continue
            unique_ratio = working_df[col].nunique(dropna=True) / max(len(working_df), 1)
            if unique_ratio <= MAX_UNIQUE_RATIO_FOR_CATEGORICAL or working_df[col].dtype == object:
                categorical_columns.append(col)

        # Compute average string length for each categorical column
        avg_string_lengths = DataProfiler._compute_avg_string_lengths(working_df, categorical_columns)

        # Mark columns as "text-heavy" — free text fields that are not useful for grouping charts.
        # Criteria: avg label length > threshold  OR  nearly every row is unique (narrative text).
        total_rows = max(len(working_df), 1)
        text_columns = []
        for col in categorical_columns:
            avg_len = avg_string_lengths.get(col, 0)
            unique_ratio = working_df[col].nunique(dropna=True) / total_rows
            if avg_len > _TEXT_AVG_LEN_THRESHOLD or (avg_len > 25 and unique_ratio > 0.5):
                text_columns.append(col)

        missing_summary = working_df.isna().sum().to_dict()
        unique_summary = {col: int(working_df[col].nunique(dropna=True)) for col in working_df.columns}

        # Collect top values and frequencies for categorical columns
        # (2–200 unique values — small enough to include in LLM prompts)
        top_values: dict = {}
        value_frequencies: dict = {}
        for col in categorical_columns:
            n_unique = unique_summary.get(col, 0)
            if 2 <= n_unique <= 200:
                counts = working_df[col].value_counts(dropna=True)
                top_values[col] = counts.index[:20].astype(str).tolist()
                value_frequencies[col] = {str(k): int(v) for k, v in counts.head(20).items()}

        return DatasetProfile(
            row_count=len(working_df),
            column_count=len(working_df.columns),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            id_like_columns=id_like_columns,
            text_columns=text_columns,
            avg_string_lengths=avg_string_lengths,
            missing_summary=missing_summary,
            unique_summary=unique_summary,
            top_values=top_values,
            value_frequencies=value_frequencies,
        )

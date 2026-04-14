from __future__ import annotations

import pandas as pd


class DataCleaner:
    """
    Pre-profiling data cleaning pipeline (adapted from best practices).
    Runs before DataProfiler so the profiler sees clean, correctly-typed data.
    """

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()

        # 1. Strip whitespace from all string columns
        for col in working.columns:
            if working[col].dtype == object:
                working[col] = working[col].str.strip()

        # 2. Convert numeric-like strings: handles $1,234 / 45% / "1,000.50"
        for col in working.columns:
            if working[col].dtype == object:
                cleaned = (
                    working[col].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                coerced = pd.to_numeric(cleaned, errors="coerce")
                success = coerced.notna().mean()
                if success >= 0.85 and coerced.notna().sum() >= 3:
                    working[col] = coerced

        # 3. Drop exact duplicate rows silently
        before = len(working)
        working = working.drop_duplicates().reset_index(drop=True)
        after = len(working)
        if before != after:
            pass  # duplicates dropped silently

        return working

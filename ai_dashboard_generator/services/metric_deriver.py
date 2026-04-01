"""
MetricDeriver
=============
Automatically detects and pre-computes derived metrics from the loaded dataset
BEFORE the LLM chat engine receives it.  The results are:

  1. New columns added to the dataframe (e.g. "duration_hours")
  2. A list of DerivedMetric descriptors injected into DatasetProfile so the
     LLM system prompt can reference them by name.

Derived metric types
--------------------
- Duration   : end_timestamp - start_timestamp  → hours  (or days if > 48 h)
- Hour-of-day: extracted from any datetime column
- Day-of-week: extracted from any datetime column
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from core.schemas import DerivedMetric


# ── Hint lists for pairing start/end timestamps ────────────────────────────

_START_HINTS = [
    "creation_date", "created_at", "checkout_date", "check_out",
    "borrow_date", "start_date", "start_time", "loan_date",
    "out_date", "issued_date", "opened_at", "begin_date",
]
_END_HINTS = [
    "return_date", "returned_at", "checkin_date", "check_in",
    "end_date", "end_time", "due_date", "return_time",
    "in_date", "closed_at", "resolved_date", "completed_date",
]


class MetricDeriver:
    """Enrich a dataframe with automatically computed metrics."""

    @staticmethod
    def derive(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[DerivedMetric]]:
        """
        Parameters
        ----------
        df : pd.DataFrame — clean, normalised dataframe from DataLoader

        Returns
        -------
        enriched_df : pd.DataFrame — same df with new derived columns appended
        metrics     : list of DerivedMetric descriptors for the system prompt
        """
        df = df.copy()
        derived: List[DerivedMetric] = []

        # ── Step 1: parse all plausible datetime columns ────────────────────
        dt_cols = MetricDeriver._find_datetime_columns(df)
        for col in dt_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # ── Step 2: compute durations from start/end pairs ──────────────────
        start_cols = [c for c in dt_cols if any(h in c.lower() for h in _START_HINTS)]
        end_cols   = [c for c in dt_cols if any(h in c.lower() for h in _END_HINTS)]

        # If no named pairs but two datetime columns exist, try them
        if not start_cols and not end_cols and len(dt_cols) >= 2:
            start_cols = [dt_cols[0]]
            end_cols   = [dt_cols[1]]

        used_pairs: set = set()
        for start_col in start_cols:
            for end_col in end_cols:
                if start_col == end_col or (start_col, end_col) in used_pairs:
                    continue
                used_pairs.add((start_col, end_col))

                metric = MetricDeriver._try_duration(df, start_col, end_col)
                if metric is not None:
                    name = metric.name
                    delta = (
                        pd.to_datetime(df[end_col], errors="coerce")
                        - pd.to_datetime(df[start_col], errors="coerce")
                    )
                    if metric.unit == "days":
                        df[name] = (delta.dt.total_seconds() / 86400).where(
                            lambda s: s >= 0
                        )
                    else:
                        df[name] = (delta.dt.total_seconds() / 3600).where(
                            lambda s: s >= 0
                        )
                    derived.append(metric)

        # ── Step 3: extract hour-of-day and day-of-week ─────────────────────
        for col in dt_cols[:3]:   # cap at 3 to avoid column explosion
            col_series = pd.to_datetime(df[col], errors="coerce")
            valid_ratio = col_series.notna().mean()
            if valid_ratio < 0.4:
                continue

            hour_col = f"{col}_hour"
            if hour_col not in df.columns:
                df[hour_col] = col_series.dt.hour
                derived.append(DerivedMetric(
                    name=hour_col,
                    description=f"Hour of day (0–23) extracted from '{col}'",
                    from_cols=[col],
                    unit="hour (0–23)",
                ))

            dow_col = f"{col}_day_of_week"
            if dow_col not in df.columns:
                df[dow_col] = col_series.dt.day_name()
                derived.append(DerivedMetric(
                    name=dow_col,
                    description=f"Day of week extracted from '{col}'",
                    from_cols=[col],
                    unit="day name",
                ))

        return df, derived

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _find_datetime_columns(df: pd.DataFrame) -> List[str]:
        """Return columns that are already datetime or parseable as datetime."""
        from utils.helpers import DATE_HINTS
        result = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                result.append(col)
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue  # don't misinterpret numeric IDs as timestamps
            col_lower = col.lower()
            is_hint = any(h in col_lower for h in DATE_HINTS)
            try:
                parsed = pd.to_datetime(df[col].head(100), errors="coerce")
                valid = parsed.notna().mean()
            except Exception:
                valid = 0.0
            if is_hint and valid > 0.5:
                result.append(col)
            elif valid > 0.8:
                result.append(col)
        return result

    @staticmethod
    def _try_duration(
        df: pd.DataFrame, start_col: str, end_col: str
    ) -> DerivedMetric | None:
        """
        Compute end - start.  Returns a DerivedMetric descriptor if at least
        30 % of rows produce a positive duration, else None.
        """
        try:
            delta = pd.to_datetime(df[end_col], errors="coerce") - pd.to_datetime(
                df[start_col], errors="coerce"
            )
            hours = delta.dt.total_seconds() / 3600
            valid_positive = hours[(hours > 0) & hours.notna()]
            coverage = len(valid_positive) / max(len(df), 1)
            if coverage < 0.3:
                return None

            median_h = valid_positive.median()
            if median_h > 48:
                unit = "days"
                col_name = "duration_days"
            else:
                unit = "hours"
                col_name = "duration_hours"

            # Avoid clobbering if column already exists from a prior pair
            if col_name in df.columns:
                col_name = f"{start_col}_to_{end_col}_{unit}"

            return DerivedMetric(
                name=col_name,
                description=(
                    f"Duration ({unit}) between '{start_col}' and '{end_col}'. "
                    f"Covers {coverage:.0%} of rows."
                ),
                from_cols=[start_col, end_col],
                unit=unit,
            )
        except Exception:
            return None

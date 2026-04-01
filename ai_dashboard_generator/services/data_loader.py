from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from utils.helpers import normalize_column_name


class DataLoader:
    @staticmethod
    def load_file(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """
        Load CSV or Excel.

        For Excel, reads ALL sheets and either:
          - combines them when they share ≥2 common columns (adds a '_sheet' column), or
          - falls back to the largest single sheet.

        Returns
        -------
        df : pd.DataFrame
            The loaded (and normalised) dataframe.
        sheet_metadata : dict
            {sheet_name: {rows, columns, dtypes}}  — empty for CSV.
        """
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df = DataLoader._normalise(df)
            DataLoader._validate(df)
            return df, {}

        if file_name.endswith((".xlsx", ".xls")):
            raw_sheets: Dict[str, pd.DataFrame] = pd.read_excel(
                uploaded_file, sheet_name=None
            )
            if not raw_sheets:
                raise ValueError("Excel file contains no readable sheets.")

            # Normalise every sheet and collect metadata
            sheets: Dict[str, pd.DataFrame] = {}
            sheet_metadata: Dict[str, dict] = {}

            for sname, sdf in raw_sheets.items():
                sdf = sdf.dropna(how="all").dropna(axis=1, how="all")
                if sdf.empty:
                    continue
                sdf = DataLoader._normalise(sdf)
                sheets[sname] = sdf
                sheet_metadata[sname] = {
                    "rows": len(sdf),
                    "columns": list(sdf.columns),
                    "dtypes": {c: str(t) for c, t in sdf.dtypes.items()},
                }

            if not sheets:
                raise ValueError("All sheets in the Excel file are empty.")

            if len(sheets) == 1:
                df = next(iter(sheets.values()))
                DataLoader._validate(df)
                return df, sheet_metadata

            # Multi-sheet: combine if schemas are compatible
            col_sets = [set(sdf.columns) for sdf in sheets.values()]
            common_cols = set.intersection(*col_sets)

            if len(common_cols) >= 2:
                dfs = []
                for sname, sdf in sheets.items():
                    sdf = sdf.copy()
                    sdf["_sheet"] = sname
                    dfs.append(sdf)
                try:
                    combined = pd.concat(dfs, ignore_index=True, sort=False)
                    DataLoader._validate(combined)
                    return combined, sheet_metadata
                except Exception:
                    pass  # fall through to largest-sheet fallback

            # Incompatible schemas — use the largest sheet
            largest_name = max(sheets, key=lambda k: len(sheets[k]))
            df = sheets[largest_name]
            DataLoader._validate(df)
            return df, sheet_metadata

        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [normalize_column_name(str(c)) for c in df.columns]
        return df

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("The uploaded file contains no data rows.")
        if len(df.columns) == 0:
            raise ValueError("The uploaded file contains no columns.")

from __future__ import annotations

import pandas as pd
import plotly.express as px

from core.schemas import ChartSpec


class SmartChartBuilder:
    """
    Safe chart rendering engine.
    Handles: filter, grouped time series, count-without-y, missing values, aggregation.
    """

    @staticmethod
    def _validate_columns(df: pd.DataFrame, spec: ChartSpec) -> None:
        for col in [spec.x, spec.y, spec.color]:
            if col and col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")

    @staticmethod
    def _apply_filter(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        """
        Apply row filter when filter_col / filter_val / filter_values are set.

        Matching strategy (tried in order):
          1. Exact string match
          2. Case-insensitive exact match
          3. Case-insensitive substring match (filter_val inside cell value)
          4. Reverse substring match (cell value inside filter_val)

        Also supports filter_values (list) for OR-matching multiple values.
        """
        if not spec.filter_col:
            return df
        if spec.filter_col not in df.columns:
            raise ValueError(f"filter_col '{spec.filter_col}' not found.")

        col_series = df[spec.filter_col].astype(str)

        # ── Multi-value OR filter ────────────────────────────────────────────
        if spec.filter_values:
            targets = [str(v).lower() for v in spec.filter_values]
            mask = col_series.str.lower().isin(targets)
            if not mask.any():
                # fallback: substring
                mask = col_series.str.lower().apply(
                    lambda cell: any(t in cell or cell in t for t in targets)
                )
            filtered = df[mask].copy()
            if filtered.empty:
                raise ValueError(
                    f"No rows match {spec.filter_col} in {spec.filter_values}."
                )
            return filtered

        # ── Single-value filter ──────────────────────────────────────────────
        if spec.filter_val is None:
            return df

        target = str(spec.filter_val)
        target_lower = target.lower()

        # 1. Exact
        mask = col_series == target
        if mask.any():
            return df[mask].copy()

        # 2. Case-insensitive exact
        mask = col_series.str.lower() == target_lower
        if mask.any():
            return df[mask].copy()

        # 3. filter_val substring inside cell
        mask = col_series.str.lower().str.contains(target_lower, regex=False, na=False)
        if mask.any():
            return df[mask].copy()

        # 4. cell substring inside filter_val
        mask = col_series.str.lower().apply(lambda cell: cell in target_lower)
        if mask.any():
            return df[mask].copy()

        raise ValueError(
            f"No rows match {spec.filter_col}={spec.filter_val!r} "
            f"(tried exact, case-insensitive, and substring matching)."
        )

    @staticmethod
    def _prepare_time_series(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        if not spec.x or not spec.y:
            raise ValueError("Time series requires x and y columns.")

        working = df.copy()
        working[spec.x] = pd.to_datetime(working[spec.x], errors="coerce")
        working[spec.y] = pd.to_numeric(working[spec.y], errors="coerce")
        working = working.dropna(subset=[spec.x, spec.y])

        if working.empty:
            raise ValueError("No valid rows for time series chart.")

        granularity = spec.time_granularity or "monthly"
        freq_map = {"daily": "D", "weekly": "W", "monthly": "M",
                    "quarterly": "Q", "yearly": "Y"}
        freq = freq_map.get(granularity, "M")
        working["_period"] = working[spec.x].dt.to_period(freq).astype(str)

        agg = spec.agg or "sum"

        # Group by period + optional color column for multi-line chart
        group_cols = ["_period"]
        if spec.color and spec.color in working.columns:
            group_cols.append(spec.color)

        if agg == "sum":
            out = working.groupby(group_cols, as_index=False)[spec.y].sum()
        elif agg == "mean":
            out = working.groupby(group_cols, as_index=False)[spec.y].mean()
        elif agg == "count":
            out = working.groupby(group_cols, as_index=False)[spec.y].count()
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")

        out = out.sort_values("_period")
        out.rename(columns={"_period": spec.x}, inplace=True)
        return out

    @staticmethod
    def _aggregate(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        if not spec.x or not spec.y:
            raise ValueError("Aggregated chart requires x and y columns.")

        working = df.dropna(subset=[spec.x, spec.y]).copy()
        working[spec.y] = pd.to_numeric(working[spec.y], errors="coerce")
        working = working.dropna(subset=[spec.y])

        if working.empty:
            raise ValueError("No valid rows after cleaning.")

        agg = spec.agg or "sum"
        if agg == "sum":
            result = working.groupby(spec.x, as_index=False)[spec.y].sum()
        elif agg == "mean":
            result = working.groupby(spec.x, as_index=False)[spec.y].mean()
        elif agg == "count":
            result = working.groupby(spec.x, as_index=False)[spec.y].count()
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")

        if spec.sort_by == "y_desc":
            result = result.sort_values(by=spec.y, ascending=False)
        if spec.top_n:
            result = result.head(spec.top_n)
        return result

    @staticmethod
    def build(df: pd.DataFrame, spec: ChartSpec):
        if df.empty:
            raise ValueError("Cannot build chart from empty dataframe.")

        # ── Special case: count time-series (records per period, no y) ──────
        if spec.chart_type in ("time_series", "line") and spec.agg == "count" and spec.y is None:
            if not spec.x:
                raise ValueError("count time series requires x column.")
            working = df.copy()
            working[spec.x] = pd.to_datetime(working[spec.x], errors="coerce")
            working = working.dropna(subset=[spec.x])
            if working.empty:
                raise ValueError("No valid datetime rows for count time series.")
            granularity = spec.time_granularity or "monthly"
            freq_map = {"daily": "D", "weekly": "W", "monthly": "M",
                        "quarterly": "Q", "yearly": "Y"}
            freq = freq_map.get(granularity, "M")
            working["_period"] = working[spec.x].dt.to_period(freq).astype(str)
            result = working.groupby("_period").size().reset_index(name="_count")
            result = result.sort_values("_period")
            result.rename(columns={"_period": spec.x}, inplace=True)
            fig = px.line(result, x=spec.x, y="_count", title=spec.title, markers=True)
            fig.update_layout(xaxis_title=spec.x, yaxis_title="Count")
            return fig

        # ── Special case: count bar chart with no y column ─────────────────
        if spec.chart_type == "bar" and spec.agg == "count" and spec.y is None:
            if not spec.x:
                raise ValueError("count chart requires x column.")
            df = SmartChartBuilder._apply_filter(df, spec)
            working = df.dropna(subset=[spec.x]).copy()
            result = working.groupby(spec.x).size().reset_index(name="_count")
            if spec.sort_by == "y_desc":
                result = result.sort_values("_count", ascending=False)
            if spec.top_n:
                result = result.head(spec.top_n)
            fig = px.bar(result, x=spec.x, y="_count", title=spec.title, text_auto=True)
            fig.update_layout(xaxis_title=spec.x, yaxis_title="Count")
            return fig

        # ── Standard path ──────────────────────────────────────────────────
        SmartChartBuilder._validate_columns(df, spec)
        df = SmartChartBuilder._apply_filter(df, spec)

        if spec.chart_type in ("time_series", "line"):
            plot_df = SmartChartBuilder._prepare_time_series(df, spec)
            color_col = spec.color if spec.color and spec.color in plot_df.columns else None
            fig = px.line(
                plot_df, x=spec.x, y=spec.y,
                color=color_col,
                title=spec.title,
                markers=True,
            )
            fig.update_layout(xaxis_title=spec.x, yaxis_title=spec.y)
            return fig

        if spec.chart_type == "bar":
            plot_df = SmartChartBuilder._aggregate(df, spec)
            fig = px.bar(plot_df, x=spec.x, y=spec.y, title=spec.title, text_auto=True)
            fig.update_layout(xaxis_title=spec.x, yaxis_title=spec.y)
            return fig

        if spec.chart_type == "histogram":
            if not spec.x:
                raise ValueError("Histogram requires x column.")
            working = df.copy()
            working[spec.x] = pd.to_numeric(working[spec.x], errors="coerce")
            working = working.dropna(subset=[spec.x])
            if working.empty:
                raise ValueError("No valid numeric values for histogram.")
            fig = px.histogram(working, x=spec.x, title=spec.title, nbins=30)
            fig.update_layout(xaxis_title=spec.x, yaxis_title="Count")
            return fig

        if spec.chart_type == "scatter":
            if not spec.x or not spec.y:
                raise ValueError("Scatter requires x and y.")
            working = df.copy()
            working[spec.x] = pd.to_numeric(working[spec.x], errors="coerce")
            working[spec.y] = pd.to_numeric(working[spec.y], errors="coerce")
            working = working.dropna(subset=[spec.x, spec.y])
            if working.empty:
                raise ValueError("No valid rows for scatter chart.")
            try:
                import statsmodels  # noqa: F401
                trendline = "ols" if len(working) > 10 else None
            except ImportError:
                trendline = None
            fig = px.scatter(working, x=spec.x, y=spec.y, title=spec.title, trendline=trendline)
            fig.update_layout(xaxis_title=spec.x, yaxis_title=spec.y)
            return fig

        if spec.chart_type == "pie":
            if not spec.x:
                raise ValueError("Pie chart requires x column (category).")
            working = df.dropna(subset=[spec.x]).copy()

            if spec.y:
                # Aggregate numeric by category
                working[spec.y] = pd.to_numeric(working[spec.y], errors="coerce")
                working = working.dropna(subset=[spec.y])
                agg = spec.agg or "sum"
                if agg == "sum":
                    result = working.groupby(spec.x, as_index=False)[spec.y].sum()
                elif agg == "mean":
                    result = working.groupby(spec.x, as_index=False)[spec.y].mean()
                else:
                    result = working.groupby(spec.x, as_index=False)[spec.y].sum()
                if spec.top_n:
                    result = result.nlargest(spec.top_n, spec.y)
                fig = px.pie(result, names=spec.x, values=spec.y, title=spec.title)
            else:
                # Count by category
                result = working[spec.x].value_counts().reset_index()
                result.columns = [spec.x, "_count"]
                if spec.top_n:
                    result = result.head(spec.top_n)
                fig = px.pie(result, names=spec.x, values="_count", title=spec.title)
            return fig

        if spec.chart_type == "heatmap":
            numeric_df = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
            if numeric_df.shape[1] < 2:
                raise ValueError("Need at least 2 numeric columns for heatmap.")
            corr = numeric_df.corr(numeric_only=True).round(2)
            fig = px.imshow(corr, text_auto=True, title=spec.title, aspect="auto")
            return fig

        raise ValueError(f"Unsupported chart_type: {spec.chart_type}")


# Backward-compatible alias
ChartBuilder = SmartChartBuilder

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from core.schemas import ChartSpec, DashboardSpec, DatasetProfile, SemanticGroup


class StoryPlanner:
    """
    Reorders and augments charts into a storytelling sequence:

    1. Overview      — metric by semantic group (broad picture)
    2. Trend         — metric over time, split by category
    3. Drill-down    — top items inside the leading category
    4. Volume        — count of items per category
    5. Distribution  — histogram of primary metric
    6. Correlation   — heatmap (carried over from base spec)

    If no semantic groups are found, the original DashboardSpec is returned unchanged.
    """

    @staticmethod
    def _primary_metric(spec: DashboardSpec) -> Optional[str]:
        for chart in spec.charts:
            if chart.y:
                return chart.y
        return None

    @staticmethod
    def _datetime_col(spec: DashboardSpec) -> Optional[str]:
        for chart in spec.charts:
            if chart.chart_type in ("time_series", "line") and chart.x:
                return chart.x
        return None

    @staticmethod
    def _time_granularity(spec: DashboardSpec) -> str:
        for chart in spec.charts:
            if chart.chart_type in ("time_series", "line") and chart.time_granularity:
                return chart.time_granularity
        return "monthly"

    @staticmethod
    def _top_category(
        df: pd.DataFrame,
        derived_col: str,
        metric_col: Optional[str],
    ) -> Optional[str]:
        """Category with the highest total of metric_col (or highest count if metric is None)."""
        if derived_col not in df.columns:
            return None
        try:
            if metric_col and metric_col in df.columns:
                numeric = pd.to_numeric(df[metric_col], errors="coerce")
                temp = df[[derived_col]].copy()
                temp["_m"] = numeric
                grouped = temp.groupby(derived_col, dropna=True)["_m"].sum()
            else:
                grouped = df[derived_col].dropna().value_counts()
            return str(grouped.idxmax()) if not grouped.empty else None
        except Exception:
            return None

    @staticmethod
    def plan(
        df: pd.DataFrame,
        profile: DatasetProfile,
        semantic_groups: List[SemanticGroup],
        existing_spec: DashboardSpec,
    ) -> DashboardSpec:
        if not semantic_groups:
            return existing_spec

        primary_metric = StoryPlanner._primary_metric(existing_spec)
        datetime_col   = StoryPlanner._datetime_col(existing_spec)
        granularity    = StoryPlanner._time_granularity(existing_spec)

        # Pick the highest-quality semantic group
        best = max(semantic_groups, key=lambda g: g.coverage * g.confidence)
        derived = best.derived_column
        src     = best.source_column

        charts: List[ChartSpec] = []

        # ── 1. Overview: metric by derived category ────────────────────────
        if primary_metric:
            charts.append(ChartSpec(
                title=f"{primary_metric} by {derived.replace('_', ' ')}",
                chart_type="bar",
                x=derived,
                y=primary_metric,
                agg="sum",
                sort_by="y_desc",
                description=(
                    f"Overview: total {primary_metric} for each "
                    f"{src} category — start here for the big picture."
                ),
            ))

        # ── 2. Trend over time, split by category ─────────────────────────
        if datetime_col and primary_metric:
            charts.append(ChartSpec(
                title=f"{primary_metric} trend by category ({granularity})",
                chart_type="time_series",
                x=datetime_col,
                y=primary_metric,
                color=derived,
                agg="sum",
                time_granularity=granularity,
                description=(
                    f"How each {src} category is trending over time — "
                    "compare growth rates side by side."
                ),
            ))

        # ── 3. Drill-down: top items in the leading category ───────────────
        top_val = StoryPlanner._top_category(df, derived, primary_metric)
        if top_val and primary_metric:
            charts.append(ChartSpec(
                title=f"Top {src} inside '{top_val}'",
                chart_type="bar",
                x=src,
                y=primary_metric,
                agg="sum",
                sort_by="y_desc",
                top_n=10,
                filter_col=derived,
                filter_val=top_val,
                description=(
                    f"Drill-down: which individual {src} values drive "
                    f"the most {primary_metric} within {top_val}?"
                ),
            ))

        # ── 4. Volume: count of items per category ─────────────────────────
        charts.append(ChartSpec(
            title=f"Number of {src} per category",
            chart_type="bar",
            x=derived,
            y=None,
            agg="count",
            sort_by="y_desc",
            description=f"How many {src} values belong to each category.",
        ))

        # ── 5 & 6. Carry over histogram, scatter, heatmap from base spec ──
        for chart in existing_spec.charts:
            if chart.chart_type in ("histogram", "scatter", "heatmap"):
                charts.append(chart)

        return DashboardSpec(kpis=existing_spec.kpis, charts=charts)

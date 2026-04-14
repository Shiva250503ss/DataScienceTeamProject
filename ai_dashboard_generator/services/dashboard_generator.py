from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.schemas import ChartSpec, DashboardSpec, DatasetProfile
from services.column_intelligence import ColumnIntelligence
from services.llm_column_intelligence import LLMColumnIntelligence
from services.semantic_grouper import SemanticGrouper
from services.story_planner import StoryPlanner


class SmartDashboardGenerator:
    """
    Generates a dashboard plan using:
    1. Statistical scoring (completeness, variance, ID penalties)
    2. LLM semantic analysis (rule-based fallback when no LLM client provided)
    3. Combined final ranking: final_score = stat * 0.50 + llm * 0.50
    """

    @staticmethod
    def _safe_sum(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce")
        return float(s.fillna(0).sum())

    @staticmethod
    def _safe_mean(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if not s.empty else 0.0

    @staticmethod
    def _safe_median(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.median()) if not s.empty else 0.0

    @staticmethod
    def _build_kpis(
        df: pd.DataFrame,
        primary_metric: Optional[str],
        ranked_numeric: List[str],
    ) -> List[Dict[str, Any]]:
        kpis: List[Dict[str, Any]] = [
            {"label": "Rows", "value": int(len(df))},
            {"label": "Columns", "value": int(len(df.columns))},
        ]

        if primary_metric:
            kpis.append({
                "label": f"Total {primary_metric}",
                "value": SmartDashboardGenerator._safe_sum(df[primary_metric]),
            })
            kpis.append({
                "label": f"Average {primary_metric}",
                "value": SmartDashboardGenerator._safe_mean(df[primary_metric]),
            })
            kpis.append({
                "label": f"Median {primary_metric}",
                "value": SmartDashboardGenerator._safe_median(df[primary_metric]),
            })

        for col in ranked_numeric:
            if col != primary_metric:
                kpis.append({
                    "label": f"Average {col}",
                    "value": SmartDashboardGenerator._safe_mean(df[col]),
                })
                break

        return kpis[:6]

    @staticmethod
    def _is_useful_numeric(df: pd.DataFrame, col: str) -> bool:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return len(s) > 1 and s.nunique() > 1

    @staticmethod
    def _is_chartable_categorical(col: str, profile: DatasetProfile) -> bool:
        """
        A categorical column is chartable when:
        - It is NOT a text-heavy free-text column (avg label length ≤ 45 chars)
        - It has 2–50 unique values (enough variation but not too many)
        """
        if col in profile.text_columns:
            return False
        n_unique = profile.unique_summary.get(col, 0)
        return 2 <= n_unique <= 50

    @staticmethod
    def generate(
        df: pd.DataFrame,
        profile: DatasetProfile,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, DashboardSpec]:
        """
        Args:
            df:         The dataset.
            profile:    DatasetProfile from DataProfiler.
            llm_client: Optional LLM client with .generate(prompt, model) -> str.
                        Pass None to use rule-based semantic fallback (default).
            llm_model:  Model identifier forwarded to llm_client.generate().
        """
        numeric_columns = profile.numeric_columns or []
        categorical_columns = profile.categorical_columns or []
        datetime_columns = profile.datetime_columns or []

        # Guard: empty dataset → return minimal spec
        if not numeric_columns and not categorical_columns and not datetime_columns:
            return df.copy(), DashboardSpec(
                kpis=[
                    {"label": "Rows", "value": int(len(df))},
                    {"label": "Columns", "value": int(len(df.columns))},
                ],
                charts=[],
            )

        # --- Step 1: LLM (or rule-based) semantic analysis ---
        llm_helper = LLMColumnIntelligence(client=llm_client, model=llm_model)
        all_columns = numeric_columns + categorical_columns + datetime_columns
        llm_results = llm_helper.analyze_columns(df, all_columns)

        # --- Step 2: Rank numeric columns with hybrid score ---
        ranked_numeric_scores = ColumnIntelligence.rank_numeric_columns(
            df=df,
            numeric_columns=numeric_columns,
            llm_results=llm_results,
        )

        # Keep all numerics sorted by score — only gate on data quality, not score threshold.
        # A score threshold was causing LLM-penalised columns to be dropped entirely,
        # leaving too few columns for scatter and heatmap charts.
        ranked_numeric = [
            item.column for item in ranked_numeric_scores
            if SmartDashboardGenerator._is_useful_numeric(df, item.column)
        ]

        primary_metric = ranked_numeric[0] if ranked_numeric else None
        secondary_metric = ranked_numeric[1] if len(ranked_numeric) > 1 else None

        # --- Step 3: Choose primary category (LLM-guided, then fallback) ---
        # Only consider chartable (non-text-heavy, reasonable cardinality) columns.
        chartable_cats = [
            c for c in categorical_columns
            if SmartDashboardGenerator._is_chartable_categorical(c, profile)
        ]
        primary_category = None
        for col in chartable_cats:
            result = llm_results.get(col)
            if result and result.semantic_type == "category" and not result.is_identifier:
                primary_category = col
                break
        if primary_category is None and chartable_cats:
            primary_category = chartable_cats[0]

        # --- Step 4: Choose best datetime (LLM-guided, then stat, then first) ---
        best_datetime = None
        for col in datetime_columns:
            result = llm_results.get(col)
            if result and result.semantic_type == "datetime":
                best_datetime = col
                break
        if best_datetime is None:
            best_datetime = ColumnIntelligence.detect_best_datetime_column(df, datetime_columns)

        # --- Step 5: Detect time granularity if we have a datetime column ---
        time_granularity = "monthly"
        if best_datetime:
            try:
                time_granularity = ColumnIntelligence.detect_time_granularity(df, best_datetime)
            except Exception:
                pass

        # --- Step 6: Build KPIs ---
        kpis = SmartDashboardGenerator._build_kpis(df, primary_metric, ranked_numeric)
        # If no numeric metric, surface category counts as KPIs instead
        if not primary_metric and chartable_cats:
            for cat_col in chartable_cats[:3]:
                kpis.append({
                    "label": f"Unique {cat_col}",
                    "value": int(df[cat_col].nunique()),
                })
            kpis = kpis[:6]

        # --- Step 7: Build chart specs ---
        charts: List[ChartSpec] = []

        if best_datetime and primary_metric:
            charts.append(ChartSpec(
                title=f"{primary_metric} trend over time ({time_granularity})",
                chart_type="time_series",
                x=best_datetime,
                y=primary_metric,
                agg="sum",
                time_granularity=time_granularity,
                description=f"Aggregated {primary_metric} over {time_granularity} periods.",
            ))

        if primary_category and primary_metric:
            charts.append(ChartSpec(
                title=f"{primary_metric} by {primary_category}",
                chart_type="bar",
                x=primary_category,
                y=primary_metric,
                agg="sum",
                top_n=12,
                sort_by="y_desc",
                description=f"Top categories ranked by total {primary_metric}.",
            ))

        if primary_metric:
            charts.append(ChartSpec(
                title=f"Distribution of {primary_metric}",
                chart_type="histogram",
                x=primary_metric,
                description=f"Distribution view for {primary_metric}.",
            ))

        if primary_metric and secondary_metric:
            charts.append(ChartSpec(
                title=f"{secondary_metric} vs {primary_metric}",
                chart_type="scatter",
                x=primary_metric,
                y=secondary_metric,
                description=f"Relationship between {primary_metric} and {secondary_metric}.",
            ))

        if len(ranked_numeric) >= 2:
            charts.append(ChartSpec(
                title="Correlation heatmap",
                chart_type="heatmap",
                description="Correlation between numeric columns.",
            ))

        # --- Always add count charts for categorical columns ----------------
        # Only use chartable columns (not free-text, not too many unique values).
        count_cats = [c for c in chartable_cats if c != primary_category][:2]
        if primary_category:
            count_cats = [primary_category] + count_cats
        for cat_col in count_cats[:3]:
            # Skip if already covered by a metric bar chart above
            already_covered = any(
                c.chart_type == "bar" and c.x == cat_col and c.y is not None
                for c in charts
            )
            if already_covered:
                continue
            charts.append(ChartSpec(
                title=f"Records per {cat_col}",
                chart_type="bar",
                x=cat_col,
                y=None,
                agg="count",
                sort_by="y_desc",
                top_n=20,
                description=f"How many records belong to each {cat_col} value.",
            ))

        # --- Fallback: count over time when no numeric metric ---------------
        if not primary_metric and best_datetime:
            charts.append(ChartSpec(
                title=f"Records over time ({time_granularity})",
                chart_type="time_series",
                x=best_datetime,
                y=None,
                agg="count",
                time_granularity=time_granularity,
                description="Number of records per time period.",
            ))

        base_spec = DashboardSpec(kpis=kpis, charts=charts)

        # --- Step 8: Semantic grouping + story ordering ---
        grouper = SemanticGrouper(client=llm_client, model=llm_model)
        enriched_df, semantic_groups = grouper.run(df, profile)
        final_spec = StoryPlanner.plan(enriched_df, profile, semantic_groups, base_spec)

        return enriched_df, final_spec


# Backward-compatible alias
DashboardGenerator = SmartDashboardGenerator

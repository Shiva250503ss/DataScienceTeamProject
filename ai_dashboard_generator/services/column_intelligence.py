from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.llm_column_intelligence import LLMColumnResult


@dataclass
class ColumnScore:
    column: str
    role: str
    semantic_type: str
    statistical_score: float
    llm_score: float
    final_score: float
    reasons: List[str]


class ColumnIntelligence:
    """
    Hybrid column scorer.

    final_score = (stat_score * stat_weight) + (llm_score * llm_weight)

    Equal weighting (50/50) between statistical and LLM/semantic scoring
    to give semantic context equal importance in KPI and feature selection.
    """

    # ------------------------------------------------------------------
    # Numeric scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score_numeric_column(
        df: pd.DataFrame,
        column: str,
        llm_result: Optional[LLMColumnResult] = None,
        stat_weight: float = 0.50,
        llm_weight: float = 0.50,
    ) -> ColumnScore:
        series = pd.to_numeric(df[column], errors="coerce")
        stat_score = 0.0
        reasons: List[str] = []

        valid = series.dropna()
        if valid.empty:
            return ColumnScore(
                column=column,
                role="numeric",
                semantic_type="unknown",
                statistical_score=-999.0,
                llm_score=0.0,
                final_score=-999.0,
                reasons=["all values invalid"],
            )

        # 1. Completeness
        missing_ratio = series.isna().mean()
        completeness_score = (1 - missing_ratio) * 20
        stat_score += completeness_score
        reasons.append(f"completeness={completeness_score:.2f}")

        # 2. Non-zero ratio
        non_zero_ratio = (valid != 0).mean()
        non_zero_score = non_zero_ratio * 15
        stat_score += non_zero_score
        reasons.append(f"non_zero={non_zero_score:.2f}")

        # 3. Spread / variance
        std = float(valid.std()) if len(valid) > 1 else 0.0
        spread_score = min(np.log1p(std), 10) if std > 0 else 0.0
        stat_score += spread_score
        reasons.append(f"spread={spread_score:.2f}")

        # 4. Penalty: ID-like numeric (almost all values unique)
        nunique_ratio = valid.nunique() / max(len(valid), 1)
        if nunique_ratio > 0.98:
            stat_score -= 20
            reasons.append("penalty: id-like numeric")

        # 5. Penalty: mostly constant
        top_freq = valid.value_counts(normalize=True).iloc[0]
        if top_freq > 0.9:
            stat_score -= 12
            reasons.append("penalty: mostly constant")

        # LLM contribution
        semantic_type = "unknown"
        llm_score = 0.0

        if llm_result:
            semantic_type = llm_result.semantic_type
            llm_score = llm_result.importance_score * 40

            if llm_result.is_identifier:
                llm_score -= 20
                reasons.append("llm penalty: identifier")

            reasons.append(f"llm semantic={llm_result.semantic_type}")
            reasons.append(f"llm importance={llm_result.importance_score:.2f}")

        final_score = (stat_score * stat_weight) + (llm_score * llm_weight)

        return ColumnScore(
            column=column,
            role="numeric",
            semantic_type=semantic_type,
            statistical_score=stat_score,
            llm_score=llm_score,
            final_score=final_score,
            reasons=reasons,
        )

    @staticmethod
    def rank_numeric_columns(
        df: pd.DataFrame,
        numeric_columns: List[str],
        llm_results: Optional[Dict[str, LLMColumnResult]] = None,
    ) -> List[ColumnScore]:
        ranked: List[ColumnScore] = []
        for col in numeric_columns:
            llm_result = llm_results.get(col) if llm_results else None
            ranked.append(
                ColumnIntelligence.score_numeric_column(df, col, llm_result)
            )
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Time-series helpers (used by chart builder)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_best_datetime_column(
        df: pd.DataFrame,
        datetime_columns: List[str],
    ) -> Optional[str]:
        if not datetime_columns:
            return None
        best_col = None
        best_score = -1.0
        for col in datetime_columns:
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                valid_ratio = converted.notna().mean()
                unique_count = converted.nunique(dropna=True)
                score = valid_ratio * 20 + min(unique_count / 10, 10)
                if score > best_score:
                    best_col = col
                    best_score = score
            except Exception:
                continue
        return best_col

    @staticmethod
    def detect_time_granularity(df: pd.DataFrame, datetime_col: str) -> str:
        dt = pd.to_datetime(df[datetime_col], errors="coerce").dropna().sort_values()
        if len(dt) < 3:
            return "monthly"
        diffs = dt.diff().dropna()
        if diffs.empty:
            return "monthly"
        median_diff = diffs.median()
        if median_diff <= pd.Timedelta(days=1):
            return "daily"
        if median_diff <= pd.Timedelta(days=7):
            return "weekly"
        if median_diff <= pd.Timedelta(days=31):
            return "monthly"
        if median_diff <= pd.Timedelta(days=92):
            return "quarterly"
        return "yearly"

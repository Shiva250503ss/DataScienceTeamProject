"""
DataInsighter — generates rich, natural-language insights from a pandas DataFrame.

Every public method returns a markdown-formatted string ready to display in chat.
Methods are pure functions that read data and return text; they never raise.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.schemas import DatasetProfile
from utils.formatting import format_number


# ── Domain detection ──────────────────────────────────────────────────────────

_DOMAIN_HINTS: Dict[str, List[str]] = {
    "restaurant":  ["restaurant", "cuisine", "meal", "food", "rating", "dining", "menu"],
    "hr":          ["employee", "department", "salary", "hire", "job", "attrition", "gender"],
    "ecommerce":   ["order", "product", "cart", "shipping", "purchase", "customer", "item"],
    "sales":       ["sales", "revenue", "deal", "lead", "pipeline", "quote", "opportunity"],
    "support":     ["ticket", "priority", "channel", "resolution", "escalat", "agent"],
    "academic":    ["student", "score", "grade", "subject", "exam", "marks", "class"],
    "finance":     ["expense", "budget", "cost", "amount", "transaction", "invoice", "payment"],
    "marketing":   ["campaign", "click", "impression", "conversion", "ctr", "roi", "ad"],
}

_DOMAIN_NAMES: Dict[str, str] = {
    "restaurant":  "restaurant & dining reviews",
    "hr":          "HR & employee data",
    "ecommerce":   "e-commerce / orders",
    "sales":       "sales pipeline / CRM",
    "support":     "customer support tickets",
    "academic":    "academic performance",
    "finance":     "financial / expense records",
    "marketing":   "marketing & campaign data",
}


def _infer_domain(profile: DatasetProfile) -> str:
    all_col_text = " ".join(
        profile.numeric_columns + profile.categorical_columns + profile.datetime_columns
    ).lower()
    best, best_score = "general", 0
    for domain, hints in _DOMAIN_HINTS.items():
        score = sum(1 for h in hints if h in all_col_text)
        if score > best_score:
            best, best_score = domain, score
    return best


# ── Main insighter class ──────────────────────────────────────────────────────

class DataInsighter:
    """
    Generates natural language insights from a dataset.
    All methods are static and return markdown strings. They never raise.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset story
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def describe_dataset(df: pd.DataFrame, profile: DatasetProfile) -> str:
        """
        Full narrative about what the dataset is and what's interesting about it.
        Used for 'what is this data about', 'describe the data', 'tell me a story'.
        """
        domain = _infer_domain(profile)
        domain_name = _DOMAIN_NAMES.get(domain, "dataset")

        lines: List[str] = []
        lines.append(f"**About this {domain_name}**\n")
        lines.append(
            f"This dataset has **{profile.row_count:,} records** "
            f"and **{profile.column_count} columns**.\n"
        )

        # --- What it contains ---
        lines.append("**What it tracks:**")
        for col in profile.categorical_columns:
            if col in profile.text_columns:
                continue
            n_unique = profile.unique_summary.get(col, 0)
            top_vals = profile.top_values.get(col, [])[:3]
            if n_unique > 50:
                continue
            if top_vals:
                sample = ", ".join(f"*{v}*" for v in top_vals)
                suffix = "..." if n_unique > 3 else ""
                lines.append(f"- **{col}**: {n_unique} unique values — {sample}{suffix}")
            else:
                lines.append(f"- **{col}**: {n_unique} unique values")

        for col in profile.numeric_columns[:4]:
            try:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    lines.append(
                        f"- **{col}**: {format_number(s.min())} – {format_number(s.max())} "
                        f"(avg: {format_number(s.mean())})"
                    )
            except Exception:
                pass

        # --- Key findings ---
        findings = DataInsighter.find_auto_insights(df, profile, n=5)
        if findings:
            lines.append("\n**Key findings:**")
            for f in findings:
                lines.append(f"- {f}")

        # --- Missing data note ---
        missing_cols = [
            k for k, v in profile.missing_summary.items()
            if isinstance(v, (int, float)) and v > 0
        ]
        if missing_cols:
            lines.append(
                f"\n*Note: {len(missing_cols)} column(s) have missing values "
                f"({', '.join(missing_cols[:3])}).*"
            )

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Auto-discover interesting facts
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def find_auto_insights(
        df: pd.DataFrame, profile: DatasetProfile, n: int = 5
    ) -> List[str]:
        """Return up to n interesting observations about the dataset."""
        insights: List[str] = []

        # 1. Dominant category value
        for col in profile.categorical_columns:
            if col in profile.text_columns:
                continue
            if not (2 <= profile.unique_summary.get(col, 0) <= 30):
                continue
            try:
                vc = df[col].value_counts(dropna=True)
                if vc.empty:
                    continue
                top_pct = vc.iloc[0] / max(vc.sum(), 1) * 100
                if top_pct >= 30:
                    insights.append(
                        f"**{col}** is dominated by *{vc.index[0]}* "
                        f"({top_pct:.0f}% of all records)"
                    )
                    break
            except Exception:
                pass

        # 2. Best/worst category by primary numeric column
        for num_col in profile.numeric_columns[:2]:
            try:
                s = pd.to_numeric(df[num_col], errors="coerce").dropna()
                if len(s) < 3:
                    continue
                for cat_col in profile.categorical_columns[:4]:
                    if profile.unique_summary.get(cat_col, 0) > 30:
                        continue
                    if cat_col in profile.text_columns:
                        continue
                    working = df.dropna(subset=[cat_col]).copy()
                    working[num_col] = pd.to_numeric(working[num_col], errors="coerce")
                    working = working.dropna(subset=[num_col])
                    grp = working.groupby(cat_col)[num_col].mean()
                    if len(grp) >= 2:
                        top_cat, bot_cat = grp.idxmax(), grp.idxmin()
                        insights.append(
                            f"Highest avg **{num_col}**: *{top_cat}* "
                            f"({format_number(grp.max())})"
                        )
                        insights.append(
                            f"Lowest avg **{num_col}**: *{bot_cat}* "
                            f"({format_number(grp.min())})"
                        )
                        break
                break
            except Exception:
                pass

        # 3. Strongest correlation between numeric columns
        num_cols_clean = []
        for c in profile.numeric_columns:
            try:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if s.nunique() > 3:
                    num_cols_clean.append(c)
            except Exception:
                pass

        if len(num_cols_clean) >= 2:
            try:
                corr_df = df[num_cols_clean].apply(pd.to_numeric, errors="coerce").dropna()
                if len(corr_df) >= 4:
                    corr = corr_df.corr()
                    mask = ~np.eye(corr.shape[0], dtype=bool)
                    corr_masked = corr.where(mask)
                    idx = corr_masked.abs().stack().idxmax()
                    r = float(corr.loc[idx[0], idx[1]])
                    if abs(r) > 0.4:
                        direction = "positive" if r > 0 else "negative"
                        strength = "strong" if abs(r) > 0.7 else "moderate"
                        insights.append(
                            f"**{idx[0]}** and **{idx[1]}** show a "
                            f"{strength} {direction} correlation (r = {r:.2f})"
                        )
            except Exception:
                pass

        # 4. Outlier detection
        for col in profile.numeric_columns[:2]:
            try:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) < 6:
                    continue
                mean, std = float(s.mean()), float(s.std())
                if std == 0:
                    continue
                n_out = int((abs(s - mean) > 2 * std).sum())
                if n_out >= 1:
                    insights.append(
                        f"**{col}** has **{n_out}** outlier(s) "
                        f"(beyond ±2 std from mean {format_number(mean)})"
                    )
                    break
            except Exception:
                pass

        return insights[:n]

    # ──────────────────────────────────────────────────────────────────────────
    # Comparison (bar chart) insight
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def describe_comparison(
        df: pd.DataFrame,
        cat_col: str,
        num_col: str,
        agg: str = "sum",
    ) -> str:
        """Rich insight for 'compare X by Y' charts."""
        try:
            working = df.dropna(subset=[cat_col, num_col]).copy()
            working[num_col] = pd.to_numeric(working[num_col], errors="coerce")
            working = working.dropna(subset=[num_col])
            if working.empty:
                return ""

            agg_fn_map = {"sum": "sum", "mean": "mean", "count": "count"}
            fn = agg_fn_map.get(agg, "sum")
            grouped = getattr(working.groupby(cat_col)[num_col], fn)()
            if grouped.empty:
                return ""

            top_cat = grouped.idxmax()
            top_val = grouped.max()
            bot_cat = grouped.idxmin()
            bot_val = grouped.min()
            total = grouped.sum() if agg == "sum" else None

            agg_label = {"sum": "total", "mean": "average", "count": "count"}.get(agg, agg)

            parts: List[str] = []
            pct_str = ""
            if total and total > 0:
                top_pct = top_val / total * 100
                pct_str = f" ({top_pct:.0f}% of the overall {agg_label})"

            parts.append(
                f"**{top_cat}** leads with the highest {agg_label} **{num_col}** "
                f"of **{format_number(top_val)}**{pct_str}."
            )
            parts.append(
                f"**{bot_cat}** is at the bottom with **{format_number(bot_val)}**."
            )
            if len(grouped) > 2:
                spread = top_val - bot_val
                parts.append(
                    f"The gap between the highest and lowest across "
                    f"{grouped.index.nunique()} categories is "
                    f"**{format_number(spread)}**."
                )

            return " ".join(parts)
        except Exception:
            return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Distribution insight
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def describe_distribution(df: pd.DataFrame, col: str) -> str:
        """Rich insight for histogram/distribution charts."""
        try:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 3:
                return ""

            mean = float(s.mean())
            median = float(s.median())
            std = float(s.std())
            skew = float(s.skew())
            p25 = float(s.quantile(0.25))
            p75 = float(s.quantile(0.75))
            n_out = int((abs(s - mean) > 2 * std).sum()) if std > 0 else 0

            if skew > 1:
                shape = "right-skewed — most values cluster low, with a long tail toward higher values"
            elif skew < -1:
                shape = "left-skewed — most values cluster high, with a long tail toward lower values"
            elif abs(skew) > 0.5:
                shape = "slightly skewed"
            else:
                shape = "roughly bell-shaped (symmetric)"

            parts = [
                f"**{col}** ranges from **{format_number(s.min())}** "
                f"to **{format_number(s.max())}**.",
                f"Mean: **{format_number(mean)}** | Median: **{format_number(median)}** "
                f"| Std dev: **{format_number(std)}**.",
                f"Distribution is {shape}.",
                f"50% of values fall between "
                f"**{format_number(p25)}** and **{format_number(p75)}**.",
            ]
            if n_out > 0:
                parts.append(
                    f"**{n_out}** potential outlier(s) detected (beyond ±2 std)."
                )
            return " ".join(parts)
        except Exception:
            return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Correlation insight
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def describe_correlation(df: pd.DataFrame, col1: str, col2: str) -> str:
        """Rich insight for scatter/correlation charts."""
        try:
            x = pd.to_numeric(df[col1], errors="coerce")
            y = pd.to_numeric(df[col2], errors="coerce")
            valid = pd.concat([x, y], axis=1).dropna()
            if len(valid) < 3:
                return ""

            r = float(valid[col1].corr(valid[col2]))
            strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
            direction = "positive" if r > 0 else "negative"

            if r > 0.7:
                interpretation = (
                    f" As **{col1}** increases, **{col2}** tends to increase as well."
                )
            elif r < -0.7:
                interpretation = (
                    f" As **{col1}** increases, **{col2}** tends to decrease."
                )
            elif abs(r) < 0.3:
                interpretation = (
                    f" There is little to no linear relationship between these two."
                )
            else:
                interpretation = ""

            return (
                f"Correlation between **{col1}** and **{col2}**: "
                f"r = **{r:.3f}** — a {strength} {direction} relationship.{interpretation}"
            )
        except Exception:
            return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Trend insight
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def describe_trend(df: pd.DataFrame, dt_col: str, num_col: str) -> str:
        """Rich insight for time-series charts."""
        try:
            working = df[[dt_col, num_col]].copy()
            working[dt_col] = pd.to_datetime(working[dt_col], errors="coerce")
            working[num_col] = pd.to_numeric(working[num_col], errors="coerce")
            working = working.dropna().sort_values(dt_col)
            if len(working) < 3:
                return ""

            first_val = float(working[num_col].iloc[0])
            last_val = float(working[num_col].iloc[-1])
            max_val = float(working[num_col].max())
            min_val = float(working[num_col].min())
            change = last_val - first_val
            pct_change = change / abs(first_val) * 100 if first_val != 0 else 0

            direction = "increased" if change > 0 else "decreased"
            magnitude = "significantly" if abs(pct_change) > 20 else "slightly"

            return (
                f"**{num_col}** has {magnitude} {direction} over the period — "
                f"from **{format_number(first_val)}** to **{format_number(last_val)}** "
                f"({'+' if change > 0 else ''}{pct_change:.1f}%). "
                f"Peak: **{format_number(max_val)}**, Low: **{format_number(min_val)}**."
            )
        except Exception:
            return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Superlative questions ("who has the highest X")
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def answer_superlative(
        lower_prompt: str,
        df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> Optional[str]:
        """
        Answer questions like:
          'who has the highest score', 'which region has the most sales',
          'what is the best product', 'who performed best'
        Returns a rich text answer, or None if can't handle it.
        """
        from services.prompt_parser import PromptParser  # lazy import to avoid circular

        is_max = any(w in lower_prompt for w in [
            "highest", "most", "best", "top", "maximum", "max",
            "largest", "biggest", "greatest", "leading", "performed best",
        ])
        is_min = any(w in lower_prompt for w in [
            "lowest", "least", "worst", "minimum", "min",
            "smallest", "fewest", "bottom", "performed worst",
        ])

        if not (is_max or is_min):
            return None

        # Find numeric column
        num_cols = PromptParser._find_columns(lower_prompt, profile.numeric_columns)
        if not num_cols:
            from utils.helpers import BUSINESS_NUMERIC_HINTS, pick_first_matching_column
            best = pick_first_matching_column(profile.numeric_columns, BUSINESS_NUMERIC_HINTS)
            num_cols = [best] if best else profile.numeric_columns[:1]
        if not num_cols:
            return None
        num_col = num_cols[0]

        # Find category column (mentioned or best candidate)
        cat_cols_mentioned = PromptParser._find_columns(lower_prompt, profile.categorical_columns)
        cat_candidates = cat_cols_mentioned or [
            c for c in profile.categorical_columns
            if 2 <= profile.unique_summary.get(c, 0) <= 30
            and c not in profile.text_columns
        ]
        if not cat_candidates:
            # Scalar answer
            try:
                s = pd.to_numeric(df[num_col], errors="coerce").dropna()
                if s.empty:
                    return None
                if is_max:
                    val = s.max()
                    return f"The maximum **{num_col}** is **{format_number(val)}**."
                else:
                    val = s.min()
                    return f"The minimum **{num_col}** is **{format_number(val)}**."
            except Exception:
                return None

        cat_col = cat_candidates[0]
        try:
            working = df.dropna(subset=[cat_col]).copy()
            working[num_col] = pd.to_numeric(working[num_col], errors="coerce")
            working = working.dropna(subset=[num_col])
            grp = working.groupby(cat_col)[num_col].mean()
            if grp.empty:
                return None

            if is_max:
                winner = grp.idxmax()
                val = grp.max()
                direction_word = "highest"
            else:
                winner = grp.idxmin()
                val = grp.min()
                direction_word = "lowest"

            n_records = int((working[cat_col] == winner).sum())

            answer = (
                f"**{winner}** has the {direction_word} average **{num_col}** "
                f"at **{format_number(val)}**"
                + (f" (across {n_records} records)" if n_records > 1 else "")
                + "."
            )

            # Top 3 ranking
            sorted_grp = grp.sort_values(ascending=not is_max)
            ranking_parts = []
            for rank, (name, v) in enumerate(sorted_grp.head(3).items(), 1):
                ranking_parts.append(f"{rank}. **{name}** — {format_number(v)}")
            if ranking_parts:
                answer += "\n\n**Full ranking:**\n" + "\n".join(ranking_parts)

            return answer
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Count/frequency answer
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def answer_count_question(
        lower_prompt: str,
        df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> Optional[str]:
        """
        Answer questions like: 'how many unique categories', 'how many students',
        'how many X are there', 'count of X'.
        """
        from services.prompt_parser import PromptParser

        cat_cols = PromptParser._find_columns(lower_prompt, profile.categorical_columns)
        num_cols = PromptParser._find_columns(lower_prompt, profile.numeric_columns)

        # "how many rows" / "how many records"
        if any(w in lower_prompt for w in ["rows", "records", "entries", "observations"]):
            return (
                f"This dataset has **{profile.row_count:,} rows** "
                f"and **{profile.column_count} columns**."
            )

        # "how many unique X" for a category
        if cat_cols:
            col = cat_cols[0]
            n = profile.unique_summary.get(col, df[col].nunique())
            top_vals = profile.top_values.get(col, [])[:5]
            answer = f"There are **{n}** unique **{col}** values."
            if top_vals:
                answer += f"\n\nTop values: {', '.join(f'*{v}*' for v in top_vals)}"
            return answer

        return None

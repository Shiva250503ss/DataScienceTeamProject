"""
PromptParser — converts natural language prompts into ChatResponse objects.

Each response contains:
  - text:   Rich, data-driven markdown explanation / answer (like ChatGPT would give)
  - charts: Zero or more ChartSpec objects to visualize alongside the answer

Works entirely rule-based — no LLM required.
"""
from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from core.schemas import ChartSpec, ChatResponse, DatasetProfile
from services.data_insights import DataInsighter
from utils.formatting import format_number


class PromptParser:

    # ── Aggregation keywords ───────────────────────────────────────────────────
    _AGG_KEYWORDS = {
        "average": "mean", "avg": "mean", "mean": "mean",
        "total": "sum", "sum": "sum",
        "count": "count", "number of": "count", "how many": "count",
        "max": "max", "maximum": "max",
        "min": "min", "minimum": "min",
        "median": "median",
    }

    # ── Chart-type trigger keywords ────────────────────────────────────────────
    _TIME_KEYWORDS = ["trend", "over time", "time series", "timeseries"]
    _TIME_GRAN_ONLY = ["monthly", "daily", "weekly", "yearly", "quarterly"]
    _BAR_KEYWORDS  = ["compare", "comparison", "bar chart", "bar graph",
                      "group by", "grouped", "by category", "breakdown"]
    _PIE_KEYWORDS  = ["pie chart", "pie graph", "proportion", "share",
                      "percentage breakdown", "composition"]
    _HIST_KEYWORDS = ["distribution", "spread", "histogram", "frequency"]
    _SCATTER_KEYWORDS = ["relation", "relationship", "correlation", "scatter",
                         " vs ", "versus", "against"]
    _HEATMAP_KEYWORDS = ["heatmap", "heat map", "correlation matrix"]

    # ── Granularity map ────────────────────────────────────────────────────────
    _GRANULARITY_MAP = {
        "daily": "daily", "day": "daily",
        "weekly": "weekly", "week": "weekly",
        "monthly": "monthly", "month": "monthly",
        "quarterly": "quarterly", "quarter": "quarterly",
        "yearly": "yearly", "year": "yearly", "annual": "yearly",
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def parse(
        prompt: str,
        profile: DatasetProfile,
        df: Optional[pd.DataFrame] = None,
    ) -> ChatResponse:
        """Parse a natural language prompt into a ChatResponse (text + charts)."""
        p = prompt.strip()
        if not p:
            return ChatResponse(text="Please type a question or chart request about your data.")

        lower = p.lower()

        # ── 1. Meta / conversational queries ──────────────────────────────────
        if PromptParser._is_about_query(lower):
            text = DataInsighter.describe_dataset(df, profile) if df is not None \
                else PromptParser._basic_summary(profile)
            return ChatResponse(text=text)

        if PromptParser._is_summary_query(lower):
            text = DataInsighter.describe_dataset(df, profile) if df is not None \
                else PromptParser._basic_summary(profile)
            return ChatResponse(text=text)

        if PromptParser._is_column_query(lower):
            return PromptParser._build_column_response(profile)

        if PromptParser._is_insights_query(lower):
            return PromptParser._build_insights_response(df, profile)

        if PromptParser._is_help_query(lower):
            return PromptParser._build_help_response(profile)

        # ── 2. Direct data questions ───────────────────────────────────────────
        if df is not None:
            # Superlative: "who has the highest X", "which X performs best"
            superlative = DataInsighter.answer_superlative(lower, df, profile)
            if superlative:
                # Find the best chart to accompany the answer
                cat_cols = PromptParser._find_columns(lower, profile.categorical_columns)
                num_cols = PromptParser._find_columns(lower, profile.numeric_columns)
                if not num_cols and profile.numeric_columns:
                    from utils.helpers import BUSINESS_NUMERIC_HINTS, pick_first_matching_column
                    best = pick_first_matching_column(profile.numeric_columns, BUSINESS_NUMERIC_HINTS)
                    num_cols = [best] if best else profile.numeric_columns[:1]
                charts = []
                if num_cols and cat_cols:
                    charts.append(ChartSpec(
                        title=f"Avg {num_cols[0]} by {cat_cols[0]}",
                        chart_type="bar",
                        x=cat_cols[0],
                        y=num_cols[0],
                        agg="mean",
                        sort_by="y_desc",
                        top_n=15,
                    ))
                elif num_cols and profile.categorical_columns:
                    best_cat = next(
                        (c for c in profile.categorical_columns
                         if 2 <= profile.unique_summary.get(c, 0) <= 30
                         and c not in profile.text_columns),
                        profile.categorical_columns[0] if profile.categorical_columns else None,
                    )
                    if best_cat:
                        charts.append(ChartSpec(
                            title=f"Avg {num_cols[0]} by {best_cat}",
                            chart_type="bar",
                            x=best_cat,
                            y=num_cols[0],
                            agg="mean",
                            sort_by="y_desc",
                            top_n=15,
                        ))
                return ChatResponse(text=superlative, charts=charts)

            # Count / unique questions
            if any(w in lower for w in ["how many", "count of", "how much", "total number"]):
                count_ans = DataInsighter.answer_count_question(lower, df, profile)
                if count_ans:
                    return ChatResponse(text=count_ans)

            # Scalar stat: "what is the total/avg X" (no category context)
            if any(w in lower for w in ["what is the", "what's the", "tell me the"]):
                agg = PromptParser._extract_agg(lower)
                num_cols = PromptParser._find_columns(lower, profile.numeric_columns)
                cat_cols = PromptParser._find_columns(lower, profile.categorical_columns)
                dt_cols  = PromptParser._find_columns(lower, profile.datetime_columns)
                if num_cols and not cat_cols and not dt_cols:
                    return PromptParser._build_stat_response(lower, agg, num_cols, profile, df)

        # ── 3. Chart requests ──────────────────────────────────────────────────
        agg = PromptParser._extract_agg(lower)
        top_n = PromptParser._extract_top_n(lower)
        granularity = PromptParser._extract_granularity(lower)

        num_cols = PromptParser._find_columns(lower, profile.numeric_columns)
        cat_cols = PromptParser._find_columns(lower, profile.categorical_columns)
        dt_cols  = PromptParser._find_columns(lower, profile.datetime_columns)

        intent = PromptParser._detect_intent(lower, num_cols, cat_cols, dt_cols, profile)

        if intent == "time_series":
            return PromptParser._build_time_series(
                num_cols, cat_cols, dt_cols, agg, granularity, top_n, profile, df
            )
        if intent == "pie":
            return PromptParser._build_pie(num_cols, cat_cols, agg, top_n, profile, df)
        if intent == "bar":
            return PromptParser._build_bar(num_cols, cat_cols, agg, top_n, profile, df, lower)
        if intent == "histogram":
            return PromptParser._build_histogram(num_cols, profile, df)
        if intent == "scatter":
            return PromptParser._build_scatter(num_cols, profile, df)
        if intent == "heatmap":
            return PromptParser._build_heatmap(profile)
        if intent == "count":
            return PromptParser._build_count(cat_cols, dt_cols, granularity, top_n, profile, df)

        return PromptParser._build_fallback(num_cols, cat_cols, dt_cols, agg, top_n, profile, df)

    # ── Backward-compat: return a single ChartSpec ─────────────────────────────
    @staticmethod
    def parse_chart(prompt: str, profile: DatasetProfile) -> ChartSpec:
        resp = PromptParser.parse(prompt, profile)
        if resp.charts:
            return resp.charts[0]
        raise ValueError(resp.text)

    # ──────────────────────────────────────────────────────────────────────────
    # Column matching
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_columns(prompt_lower: str, candidates: List[str]) -> List[str]:
        """Find column names mentioned in the prompt (longest match first)."""
        found = []
        for col in sorted(candidates, key=len, reverse=True):
            col_pattern = col.lower().replace("_", "[ _]")
            if re.search(r'\b' + col_pattern + r'\b', prompt_lower):
                found.append(col)
            elif col.lower().replace("_", " ") in prompt_lower:
                if col not in found:
                    found.append(col)
            elif col.lower() in prompt_lower:
                if col not in found:
                    found.append(col)
        return found

    # ──────────────────────────────────────────────────────────────────────────
    # Extraction helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_agg(lower: str) -> Optional[str]:
        for keyword, agg in PromptParser._AGG_KEYWORDS.items():
            if keyword in lower:
                return agg
        return None

    @staticmethod
    def _extract_top_n(lower: str) -> Optional[int]:
        m = re.search(r'top\s*(\d+)', lower)
        if m:
            return int(m.group(1))
        m = re.search(r'(\d+)\s*(?:best|worst|highest|lowest|largest|smallest)', lower)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _extract_granularity(lower: str) -> Optional[str]:
        for keyword, gran in PromptParser._GRANULARITY_MAP.items():
            if keyword in lower:
                return gran
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Intent detection
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_intent(
        lower: str,
        num_cols: List[str],
        cat_cols: List[str],
        dt_cols: List[str],
        profile: DatasetProfile,
    ) -> str:
        if any(k in lower for k in PromptParser._HEATMAP_KEYWORDS):
            return "heatmap"
        if any(k in lower for k in PromptParser._PIE_KEYWORDS):
            return "pie"
        if any(k in lower for k in PromptParser._SCATTER_KEYWORDS):
            return "scatter"
        if any(k in lower for k in PromptParser._HIST_KEYWORDS):
            return "histogram"
        if any(k in lower for k in PromptParser._TIME_KEYWORDS):
            return "time_series"

        if any(k in lower for k in ["how many", "count of", "number of", "count by", "counts"]):
            if not num_cols:
                return "count"

        if any(k in lower for k in PromptParser._BAR_KEYWORDS):
            return "bar"
        if re.search(r'top\s*\d+', lower) and cat_cols:
            return "bar"
        if " by " in lower and cat_cols and num_cols:
            return "bar"

        if any(k in lower for k in ["what is", "what's", "tell me", "show me the"]):
            if num_cols and not cat_cols and not dt_cols:
                return "stat_answer"

        # Granularity-only (monthly, daily, …) → time_series ONLY when not part of a column name
        for k in PromptParser._TIME_GRAN_ONLY:
            if k in lower:
                is_standalone = not any(k in c.lower() for c in (num_cols + cat_cols))
                if is_standalone and (dt_cols or profile.datetime_columns):
                    return "time_series"

        if num_cols and cat_cols:
            return "bar"
        if dt_cols and num_cols:
            return "time_series"
        if cat_cols and not num_cols:
            return "count"
        if num_cols and len(num_cols) >= 2:
            return "scatter"
        if num_cols:
            return "histogram"

        return "bar"

    # ──────────────────────────────────────────────────────────────────────────
    # Query-type classifiers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_about_query(lower: str) -> bool:
        """'What is this data about', 'what is there in data', etc."""
        patterns = [
            "what is there", "what's there", "what is in the data",
            "what is the data about", "what is this data",
            "what does this data", "tell me about this",
            "tell me about the data", "what kind of data",
            "about the data", "about this dataset",
            "what data is", "explain the data", "explain this data",
        ]
        return any(p in lower for p in patterns)

    @staticmethod
    def _is_summary_query(lower: str) -> bool:
        patterns = [
            "summary", "summarize", "describe the data", "describe dataset",
            "overview", "tell me about", "what does this data",
            "what's in this", "data summary",
        ]
        return any(p in lower for p in patterns)

    @staticmethod
    def _is_column_query(lower: str) -> bool:
        patterns = [
            "what columns", "which columns", "list columns", "column names",
            "what fields", "show columns", "what are the columns",
        ]
        return any(p in lower for p in patterns)

    @staticmethod
    def _is_insights_query(lower: str) -> bool:
        patterns = [
            "give me insights", "key insights", "interesting findings",
            "what's interesting", "what is interesting", "findings",
            "analyze the data", "analyse", "key takeaways",
            "what can you tell", "tell me something interesting",
            "show insights", "generate insights",
        ]
        return any(p in lower for p in patterns)

    @staticmethod
    def _is_help_query(lower: str) -> bool:
        phrase_patterns = [
            "what can", "what we can", "what do", "how can", "how do",
            "what should", "what to ask", "what to do",
            "capabilities", "options", "what kind",
        ]
        if any(p in lower for p in phrase_patterns):
            return True
        words = set(re.split(r'\W+', lower))
        return bool(words & {"help", "hi", "hello", "hey"})

    # ──────────────────────────────────────────────────────────────────────────
    # Smart column selectors
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_numeric(profile: DatasetProfile) -> Optional[str]:
        from utils.helpers import pick_first_matching_column, BUSINESS_NUMERIC_HINTS
        return (
            pick_first_matching_column(profile.numeric_columns, BUSINESS_NUMERIC_HINTS)
            or (profile.numeric_columns[0] if profile.numeric_columns else None)
        )

    @staticmethod
    def _pick_category(profile: DatasetProfile) -> Optional[str]:
        # Prefer chartable (non-text, 2-50 unique)
        for col in profile.categorical_columns:
            if col not in profile.text_columns and 2 <= profile.unique_summary.get(col, 0) <= 50:
                return col
        return profile.categorical_columns[0] if profile.categorical_columns else None

    @staticmethod
    def _pick_datetime(profile: DatasetProfile) -> Optional[str]:
        return profile.datetime_columns[0] if profile.datetime_columns else None

    # ──────────────────────────────────────────────────────────────────────────
    # Response builders
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _basic_summary(profile: DatasetProfile) -> str:
        lines = [
            "**Dataset Overview**",
            f"- **Rows:** {profile.row_count:,}",
            f"- **Columns:** {profile.column_count}",
        ]
        if profile.numeric_columns:
            lines.append(f"- **Numeric:** {', '.join(profile.numeric_columns)}")
        if profile.categorical_columns:
            lines.append(f"- **Categorical:** {', '.join(profile.categorical_columns)}")
        if profile.datetime_columns:
            lines.append(f"- **Datetime:** {', '.join(profile.datetime_columns)}")
        return "\n".join(lines)

    @staticmethod
    def _build_column_response(profile: DatasetProfile) -> ChatResponse:
        all_cols = (
            profile.numeric_columns + profile.categorical_columns
            + profile.datetime_columns + profile.id_like_columns
        )
        lines = [f"**Columns in this dataset ({len(all_cols)} total):**"]
        if profile.numeric_columns:
            lines.append(f"- **Numeric** ({len(profile.numeric_columns)}): "
                         f"{', '.join(profile.numeric_columns)}")
        if profile.categorical_columns:
            lines.append(f"- **Categorical** ({len(profile.categorical_columns)}): "
                         f"{', '.join(profile.categorical_columns)}")
        if profile.datetime_columns:
            lines.append(f"- **Datetime** ({len(profile.datetime_columns)}): "
                         f"{', '.join(profile.datetime_columns)}")
        if profile.id_like_columns:
            lines.append(f"- **ID-like** ({len(profile.id_like_columns)}): "
                         f"{', '.join(profile.id_like_columns)}")
        return ChatResponse(text="\n".join(lines))

    @staticmethod
    def _build_insights_response(
        df: Optional[pd.DataFrame], profile: DatasetProfile
    ) -> ChatResponse:
        """Return auto-discovered insights + a set of supporting charts."""
        if df is None:
            return ChatResponse(text=PromptParser._basic_summary(profile))

        findings = DataInsighter.find_auto_insights(df, profile, n=6)
        lines = ["**Key insights from your dataset:**\n"]
        if findings:
            for f in findings:
                lines.append(f"- {f}")
        else:
            lines.append("*No strong patterns detected — try asking specific questions.*")

        # Add supporting charts
        charts: List[ChartSpec] = []
        num_col = PromptParser._pick_numeric(profile)
        cat_col = PromptParser._pick_category(profile)
        dt_col  = PromptParser._pick_datetime(profile)

        if num_col and cat_col:
            charts.append(ChartSpec(
                title=f"Avg {num_col} by {cat_col}",
                chart_type="bar", x=cat_col, y=num_col,
                agg="mean", sort_by="y_desc", top_n=15,
            ))
        if len(profile.numeric_columns) >= 2:
            charts.append(ChartSpec(
                title="Correlation heatmap",
                chart_type="heatmap",
            ))
        if num_col:
            charts.append(ChartSpec(
                title=f"Distribution of {num_col}",
                chart_type="histogram", x=num_col,
            ))

        return ChatResponse(text="\n".join(lines), charts=charts)

    @staticmethod
    def _build_help_response(profile: DatasetProfile) -> ChatResponse:
        num = profile.numeric_columns[0] if profile.numeric_columns else "value"
        cat = profile.categorical_columns[0] if profile.categorical_columns else "category"
        dt  = profile.datetime_columns[0] if profile.datetime_columns else None

        lines = ["Here's what you can ask me:\n"]
        lines.append("**About the data:**")
        lines.append("- *What is this data about?*")
        lines.append("- *Give me insights*")
        lines.append("- *What columns are there?*")
        lines.append("- *Describe the data*")

        if dt and profile.numeric_columns:
            lines.append("\n**Trends over time:**")
            lines.append(f"- *Show monthly {num} trend*")
            lines.append(f"- *Daily {num} over time*")

        if profile.numeric_columns and profile.categorical_columns:
            lines.append("\n**Comparisons:**")
            lines.append(f"- *Compare {cat} by {num}*")
            lines.append(f"- *Top 5 {cat} by {num}*")
            lines.append(f"- *Average {num} by {cat}*")
            lines.append(f"- *Who has the highest {num}?*")

        if profile.numeric_columns:
            lines.append("\n**Distributions:**")
            lines.append(f"- *Distribution of {num}*")
            lines.append(f"- *What is the average/total {num}?*")

        if len(profile.numeric_columns) >= 2:
            n2 = profile.numeric_columns[1]
            lines.append("\n**Relationships:**")
            lines.append(f"- *Relationship between {num} and {n2}*")
            lines.append("- *Correlation matrix*")

        if profile.categorical_columns:
            lines.append("\n**Counts & Pie charts:**")
            lines.append(f"- *How many records per {cat}?*")
            lines.append(f"- *Pie chart of {cat}*")

        return ChatResponse(text="\n".join(lines))

    @staticmethod
    def _build_stat_response(
        lower: str,
        agg: Optional[str],
        num_cols: List[str],
        profile: DatasetProfile,
        df: pd.DataFrame,
    ) -> ChatResponse:
        col = num_cols[0] if num_cols else PromptParser._pick_numeric(profile)
        if not col:
            return ChatResponse(text="No numeric column found to compute statistics.")

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return ChatResponse(text=f"No valid numeric values in **{col}**.")

        agg = agg or "sum"
        stat_map = {
            "sum":    ("Total",   s.sum()),
            "mean":   ("Average", s.mean()),
            "count":  ("Count",   s.count()),
            "max":    ("Maximum", s.max()),
            "min":    ("Minimum", s.min()),
            "median": ("Median",  s.median()),
        }
        label, value = stat_map.get(agg, ("Total", s.sum()))

        # Build distribution insight
        dist_text = DataInsighter.describe_distribution(df, col)
        text = f"**{label} {col}:** {format_number(value)}"
        if dist_text:
            text += f"\n\n{dist_text}"

        chart = ChartSpec(title=f"Distribution of {col}", chart_type="histogram", x=col)
        return ChatResponse(text=text, charts=[chart])

    @staticmethod
    def _build_time_series(
        num_cols, cat_cols, dt_cols, agg, granularity, top_n,
        profile: DatasetProfile, df: Optional[pd.DataFrame],
    ) -> ChatResponse:
        dt_col  = dt_cols[0]  if dt_cols  else PromptParser._pick_datetime(profile)
        num_col = num_cols[0] if num_cols else PromptParser._pick_numeric(profile)

        if not dt_col:
            return ChatResponse(
                text="No datetime column found for a trend chart. "
                     f"Available: {', '.join(profile.numeric_columns + profile.categorical_columns)}"
            )

        granularity = granularity or "monthly"

        if not num_col or agg == "count":
            chart = ChartSpec(
                title=f"Records over time ({granularity})",
                chart_type="time_series", x=dt_col, y=None,
                agg="count", time_granularity=granularity,
            )
            return ChatResponse(
                text=f"Showing record count over time by **{granularity}** using **{dt_col}**.",
                charts=[chart],
            )

        agg = agg or "sum"
        color = cat_cols[0] if cat_cols else None
        chart = ChartSpec(
            title=f"{num_col} trend over time ({granularity})",
            chart_type="time_series", x=dt_col, y=num_col,
            color=color, agg=agg, time_granularity=granularity,
        )

        trend_text = DataInsighter.describe_trend(df, dt_col, num_col) if df is not None else ""
        text = f"Showing **{agg}** of **{num_col}** over time ({granularity})"
        if color:
            text += f", grouped by **{color}**"
        text += "."
        if trend_text:
            text += f"\n\n{trend_text}"

        return ChatResponse(text=text, charts=[chart])

    @staticmethod
    def _build_pie(
        num_cols, cat_cols, agg, top_n,
        profile: DatasetProfile, df: Optional[pd.DataFrame],
    ) -> ChatResponse:
        cat_col = cat_cols[0] if cat_cols else PromptParser._pick_category(profile)
        num_col = num_cols[0] if num_cols else None
        if not cat_col:
            return ChatResponse(text="No categorical column found for a pie chart.")

        top_n = top_n or 10
        if num_col:
            agg = agg or "sum"
            chart = ChartSpec(
                title=f"{num_col} by {cat_col}",
                chart_type="pie", x=cat_col, y=num_col, agg=agg, top_n=top_n,
            )
            text = f"Showing **{agg}** of **{num_col}** by **{cat_col}** as a pie chart."
            if df is not None:
                insight = DataInsighter.describe_comparison(df, cat_col, num_col, agg)
                if insight:
                    text += f"\n\n{insight}"
        else:
            chart = ChartSpec(
                title=f"Distribution of {cat_col}",
                chart_type="pie", x=cat_col, y=None, agg="count", top_n=top_n,
            )
            text = f"Showing count distribution of **{cat_col}** as a pie chart."
            if df is not None:
                vc = df[cat_col].value_counts(dropna=True)
                if not vc.empty:
                    top_pct = vc.iloc[0] / vc.sum() * 100
                    text += (
                        f"\n\n**{vc.index[0]}** is the most common "
                        f"({top_pct:.0f}% of all records)."
                    )
        return ChatResponse(text=text, charts=[chart])

    @staticmethod
    def _build_bar(
        num_cols, cat_cols, agg, top_n,
        profile: DatasetProfile, df: Optional[pd.DataFrame], lower: str = "",
    ) -> ChatResponse:
        cat_col = cat_cols[0] if cat_cols else PromptParser._pick_category(profile)
        num_col = num_cols[0] if num_cols else PromptParser._pick_numeric(profile)

        if not cat_col:
            return ChatResponse(text="No categorical column found for a comparison.")

        top_n = top_n or 15

        if not num_col or agg == "count":
            chart = ChartSpec(
                title=f"Count by {cat_col}", chart_type="bar",
                x=cat_col, y=None, agg="count",
                top_n=top_n, sort_by="y_desc",
            )
            text = f"Showing record count by **{cat_col}**."
            if df is not None:
                n_unique = df[cat_col].nunique()
                vc = df[cat_col].value_counts(dropna=True)
                if not vc.empty:
                    text += (
                        f" ({n_unique} unique values; "
                        f"most common: **{vc.index[0]}** with {vc.iloc[0]:,} records)"
                    )
            return ChatResponse(text=text, charts=[chart])

        agg = agg or "sum"
        chart = ChartSpec(
            title=f"{agg.title()} of {num_col} by {cat_col}",
            chart_type="bar", x=cat_col, y=num_col,
            agg=agg, top_n=top_n, sort_by="y_desc",
        )

        insight = DataInsighter.describe_comparison(df, cat_col, num_col, agg) \
            if df is not None else ""
        text = f"Showing **{agg}** of **{num_col}** by **{cat_col}**."
        if insight:
            text += f"\n\n{insight}"

        return ChatResponse(text=text, charts=[chart])

    @staticmethod
    def _build_histogram(
        num_cols, profile: DatasetProfile, df: Optional[pd.DataFrame],
    ) -> ChatResponse:
        num_col = num_cols[0] if num_cols else PromptParser._pick_numeric(profile)
        if not num_col:
            return ChatResponse(text="No numeric column found for a distribution chart.")

        chart = ChartSpec(
            title=f"Distribution of {num_col}",
            chart_type="histogram", x=num_col,
        )
        insight = DataInsighter.describe_distribution(df, num_col) if df is not None else ""
        text = f"Showing distribution of **{num_col}**."
        if insight:
            text += f"\n\n{insight}"
        return ChatResponse(text=text, charts=[chart])

    @staticmethod
    def _build_scatter(
        num_cols, profile: DatasetProfile, df: Optional[pd.DataFrame],
    ) -> ChatResponse:
        if len(num_cols) >= 2:
            x_col, y_col = num_cols[0], num_cols[1]
        elif len(num_cols) == 1 and len(profile.numeric_columns) >= 2:
            x_col = num_cols[0]
            y_col = next((c for c in profile.numeric_columns if c != x_col), None)
        elif len(profile.numeric_columns) >= 2:
            x_col, y_col = profile.numeric_columns[0], profile.numeric_columns[1]
        else:
            return ChatResponse(
                text="Need at least 2 numeric columns for a scatter/relationship chart."
            )

        chart = ChartSpec(
            title=f"{x_col} vs {y_col}",
            chart_type="scatter", x=x_col, y=y_col,
        )
        insight = DataInsighter.describe_correlation(df, x_col, y_col) \
            if df is not None else ""
        text = f"Showing relationship between **{x_col}** and **{y_col}**."
        if insight:
            text += f"\n\n{insight}"
        return ChatResponse(text=text, charts=[chart])

    @staticmethod
    def _build_heatmap(profile: DatasetProfile) -> ChatResponse:
        if len(profile.numeric_columns) < 2:
            return ChatResponse(text="Need at least 2 numeric columns for a heatmap.")
        chart = ChartSpec(title="Correlation Heatmap", chart_type="heatmap")
        return ChatResponse(
            text=f"Showing correlation heatmap across "
                 f"{len(profile.numeric_columns)} numeric columns: "
                 f"{', '.join(profile.numeric_columns)}.",
            charts=[chart],
        )

    @staticmethod
    def _build_count(
        cat_cols, dt_cols, granularity, top_n,
        profile: DatasetProfile, df: Optional[pd.DataFrame],
    ) -> ChatResponse:
        if dt_cols:
            dt_col = dt_cols[0]
            granularity = granularity or "monthly"
            chart = ChartSpec(
                title=f"Record count over time ({granularity})",
                chart_type="time_series", x=dt_col, y=None,
                agg="count", time_granularity=granularity,
            )
            return ChatResponse(
                text=f"Showing record count over time ({granularity}) using **{dt_col}**.",
                charts=[chart],
            )

        cat_col = cat_cols[0] if cat_cols else PromptParser._pick_category(profile)
        if not cat_col:
            if profile.datetime_columns:
                dt_col = profile.datetime_columns[0]
                chart = ChartSpec(
                    title="Record count over time",
                    chart_type="time_series", x=dt_col, y=None,
                    agg="count", time_granularity="monthly",
                )
                return ChatResponse(text="Showing record count over time.", charts=[chart])
            return ChatResponse(
                text="No categorical or datetime column found for a count chart."
            )

        top_n = top_n or 20
        chart = ChartSpec(
            title=f"Count by {cat_col}", chart_type="bar",
            x=cat_col, y=None, agg="count",
            top_n=top_n, sort_by="y_desc",
        )
        text = f"Showing record count by **{cat_col}**."
        if df is not None:
            vc = df[cat_col].value_counts(dropna=True)
            if not vc.empty:
                n = df[cat_col].nunique()
                text += (
                    f"\n\n**{n}** unique values. "
                    f"Most frequent: **{vc.index[0]}** ({vc.iloc[0]:,} records)."
                )
        return ChatResponse(text=text, charts=[chart])

    @staticmethod
    def _build_fallback(
        num_cols, cat_cols, dt_cols, agg, top_n,
        profile: DatasetProfile, df: Optional[pd.DataFrame],
    ) -> ChatResponse:
        if num_cols and cat_cols:
            return PromptParser._build_bar(num_cols, cat_cols, agg, top_n, profile, df)
        if num_cols and (dt_cols or profile.datetime_columns):
            return PromptParser._build_time_series(
                num_cols, cat_cols, dt_cols, agg, None, top_n, profile, df
            )
        if cat_cols:
            return PromptParser._build_count(cat_cols, dt_cols, None, top_n, profile, df)
        if num_cols:
            return PromptParser._build_histogram(num_cols, profile, df)

        num_col = PromptParser._pick_numeric(profile)
        cat_col = PromptParser._pick_category(profile)
        if num_col and cat_col:
            return PromptParser._build_bar([num_col], [cat_col], agg, top_n, profile, df)
        if num_col:
            return PromptParser._build_histogram([num_col], profile, df)
        if cat_col:
            return PromptParser._build_count([cat_col], dt_cols, None, top_n, profile, df)

        return ChatResponse(
            text=(
                "I'm not sure what to show for that. Try:\n"
                "- *What is this data about?*\n"
                "- *Give me insights*\n"
                "- *Compare [category] by [metric]*\n"
                "- *Distribution of [column]*\n"
                "- *Top 5 [category] by [metric]*"
            )
        )

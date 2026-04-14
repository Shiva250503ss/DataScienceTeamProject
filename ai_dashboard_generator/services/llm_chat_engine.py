"""
LLMChatEngine
=============
Sends user questions + rich dataset context to the LLM (Groq) and
returns ChatResponse objects containing markdown text + renderable ChartSpec
objects.

Key design decisions
---------------------
1.  FULL column catalog  — every column's type, all unique values for low-
    cardinality columns, and derived-metric descriptions are in the prompt.
2.  Filter awareness    — the LLM is explicitly taught to use filter_col /
    filter_val / filter_values in chart JSON so it can answer "charger usage"
    by filtering rows where resource = 'Charger' rather than failing.
3.  Derived metrics     — if MetricDeriver pre-computed "duration_hours", the
    prompt tells the LLM that column exists and what it represents.
4.  Semantic pre-processing  — before every LLM call, we scan the user query
    for entity mentions (item names, category values) and annotate the message
    with the exact column + matching values from the real data, so the LLM
    never needs to guess column names.
5.  Fuzzy column resolution — case-insensitive + underscore/space normalised
    matching when the LLM returns a column name that is slightly off.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.schemas import ChartSpec, ChatResponse, DatasetProfile, DerivedMetric


# ── Business-term synonym dictionary ────────────────────────────────────────
# Maps common user words → possible column-name substrings or value keywords.
# Used during semantic pre-processing to surface relevant columns/values.

_TERM_SYNONYMS: Dict[str, List[str]] = {
    # Duration / time concepts
    "duration":      ["duration", "time", "hours", "days", "minutes", "length",
                      "period", "span", "checkout", "borrow"],
    "usage":         ["usage", "use", "count", "frequency", "total", "utilization",
                      "borrowed", "checked_out", "duration"],
    "trend":         ["date", "time", "month", "week", "year", "period"],
    "average":       ["mean", "avg", "average"],
    # Common entity types
    "charger":       ["charger", "charge", "power adapter", "usb", "cable"],
    "laptop":        ["laptop", "notebook", "computer", "macbook", "thinkpad"],
    "monitor":       ["monitor", "screen", "display", "hdmi"],
    "projector":     ["projector", "beamer", "presentation"],
    "camera":        ["camera", "canon", "nikon", "gopro", "dslr"],
    "phone":         ["phone", "mobile", "iphone", "android", "smartphone"],
    "tablet":        ["tablet", "ipad", "surface"],
    "keyboard":      ["keyboard", "mouse", "input"],
    # Business metrics
    "revenue":       ["revenue", "sales", "income", "gmv", "turnover"],
    "profit":        ["profit", "margin", "earnings", "net"],
    "cost":          ["cost", "expense", "spend", "cogs", "price"],
    "quantity":      ["quantity", "qty", "units", "volume", "count", "items"],
    "customer":      ["customer", "client", "user", "member", "account"],
    "employee":      ["employee", "staff", "worker", "person", "name"],
    "department":    ["department", "dept", "team", "division", "unit"],
    "category":      ["category", "type", "kind", "class", "group", "segment"],
    "region":        ["region", "area", "zone", "location", "city", "state",
                      "country", "territory"],
    "product":       ["product", "item", "sku", "resource", "asset", "device",
                      "equipment"],
    "status":        ["status", "state", "stage", "condition", "flag"],
    "rating":        ["rating", "score", "grade", "rank", "satisfaction", "nps"],
}

# Generic / ambiguous columns — mention them in the prompt with a warning
_AMBIGUOUS_COLUMN_NAMES = {
    "value", "amount", "count", "metric", "number", "data",
    "result", "total", "qty", "score",
}


# ── Main system prompt template ─────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior data analyst AI embedded in a dashboard application.
You have COMPLETE access to the dataset described below.

Your job is to answer questions accurately, derive insights, produce charts,
and behave like a REAL analyst — not a rigid schema-matching engine.

════════════════════════════════════════════════════════════════
DATASET: {file_hint}
════════════════════════════════════════════════════════════════
Rows   : {rows:,}
Columns: {n_cols}

{sheet_section}\
────────────────────────────────────────────────────────────────
COLUMN CATALOG
────────────────────────────────────────────────────────────────
{column_catalog}

────────────────────────────────────────────────────────────────
NUMERIC STATISTICS
────────────────────────────────────────────────────────────────
{stats}

────────────────────────────────────────────────────────────────
SAMPLE ROWS (first {sample_n})
────────────────────────────────────────────────────────────────
{sample}

{derived_section}\
════════════════════════════════════════════════════════════════
ANALYST RULES — READ CAREFULLY
════════════════════════════════════════════════════════════════

1. SEMANTIC UNDERSTANDING
   • Map user words to actual columns/values intelligently.
   • "charger" → filter the '{resource_hint}' column for rows containing
     charger-related values (e.g. "Charger", "USB Charger", "Power Adapter").
   • "duration" → use the 'duration_hours' or 'duration_days' derived column
     if available, otherwise derive it from available timestamp columns.
   • "usage" → usually means count of records, total duration, or sum of a
     value column — choose the most sensible interpretation.
   • NEVER say "column not found" when you can reasonably infer the intent.

2. FILTER-BASED CHARTS
   • When the user asks about a SPECIFIC item/category (e.g. "charger",
     "laptop", "Marketing department"), use filter_col + filter_val in the
     chart JSON to restrict rows.
   • filter_val must be one of the EXACT values shown in the column catalog.
   • Example: user asks "charger usage by month"
     → chart JSON: {{"chart_type":"time_series","x":"creation_date",
       "y":null,"agg":"count","filter_col":"resource",
       "filter_val":"Charger","title":"Charger Usage by Month"}}

3. DERIVED METRICS
   • If 'duration_hours' or 'duration_days' columns exist (listed in the
     Derived Metrics section), USE THEM directly as x or y in charts.
   • Example: "average charger duration"
     → filter resource="Charger", y="duration_hours", agg="mean"

4. MULTIPLE CHARTS
   • Include up to 3 <chart> blocks when the question calls for it.
   • Each block must be valid JSON inside <chart>…</chart> tags.

5. CHART JSON SCHEMA
   <chart>{{
     "chart_type" : "bar|histogram|scatter|pie|time_series|heatmap",
     "x"          : "exact_column_name",
     "y"          : "exact_column_name_or_null",
     "agg"        : "sum|mean|count|max|min",
     "title"      : "Descriptive human-readable title",
     "top_n"      : 15,
     "filter_col" : "column_to_filter_on_or_null",
     "filter_val" : "exact_value_from_column_catalog_or_null"
   }}</chart>

   • Set y to null for count and histogram charts.
   • Set filter_col / filter_val to null when no row filter is needed.
   • Only use column names that appear in the COLUMN CATALOG above.

6. RESPONSE STYLE
   • Write like a senior analyst: interpret, explain assumptions, give numbers.
   • Use markdown: **bold**, bullet points, tables where helpful.
   • State assumptions briefly: "I filtered to Charger-type resources and
     used 'duration_hours' as the duration metric."
   • Do NOT wrap your whole response in JSON.
   • Append <chart> blocks after your text answer.

7. AMBIGUOUS QUERIES
   • If the question is genuinely ambiguous, choose the most sensible
     interpretation, answer it, then offer one alternative at the end.
   • Only ask a follow-up question if there is truly no reasonable default.
"""


class LLMChatEngine:

    # ── Public API ───────────────────────────────────────────────────────────

    @staticmethod
    def chat(
        user_prompt: str,
        df: pd.DataFrame,
        profile: DatasetProfile,
        llm_client: Any,
        llm_model: Optional[str] = None,
        history: Optional[List[dict]] = None,
        file_name: str = "uploaded dataset",
    ) -> ChatResponse:
        """Send user question + full dataset context → ChatResponse."""

        system_prompt = LLMChatEngine._build_system_prompt(df, profile, file_name)

        # Semantic pre-processing: annotate the user query with matched entities
        annotated_prompt = LLMChatEngine._annotate_query(user_prompt, df, profile)

        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        # Last 8 conversation turns for context
        if history:
            for msg in history[-8:]:
                role = msg.get("role")
                text = msg.get("text", "")
                if role in ("user", "assistant") and text:
                    messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": annotated_prompt})

        raw_response = llm_client.chat_completion(messages, model=llm_model)

        charts = LLMChatEngine._extract_charts(raw_response, profile, df)
        clean_text = re.sub(r"<chart>.*?</chart>", "", raw_response, flags=re.DOTALL).strip()

        return ChatResponse(text=clean_text, charts=charts)

    # ── System prompt builder ────────────────────────────────────────────────

    @staticmethod
    def _build_system_prompt(
        df: pd.DataFrame,
        profile: DatasetProfile,
        file_name: str = "uploaded dataset",
    ) -> str:

        all_cols = set(df.columns)

        # ── Sheet section (only for multi-sheet Excel) ──────────────────────
        sheet_section = ""
        if profile.sheet_metadata:
            lines = ["EXCEL SHEETS\n"]
            for sname, smeta in profile.sheet_metadata.items():
                lines.append(
                    f"  • '{sname}': {smeta['rows']:,} rows, "
                    f"columns: {', '.join(smeta['columns'])}"
                )
            sheet_section = "\n".join(lines) + "\n\n"

        # ── Column catalog ───────────────────────────────────────────────────
        catalog_lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = profile.unique_summary.get(col, df[col].nunique())
            missing_pct = (
                profile.missing_summary.get(col, 0) / max(len(df), 1) * 100
            )

            # Classify column
            if col in profile.datetime_columns:
                kind = "datetime"
            elif col in profile.numeric_columns:
                kind = "numeric"
            elif col in profile.id_like_columns:
                kind = "identifier"
            elif col in profile.text_columns:
                kind = "free-text"
            else:
                kind = "categorical"

            # Ambiguous column warning
            ambiguous_note = ""
            if col.lower() in _AMBIGUOUS_COLUMN_NAMES:
                ambiguous_note = " ⚠ ambiguous name — interpret based on context"

            # Show ALL unique values for low-cardinality columns (≤ 80)
            values_str = ""
            if kind == "categorical" and n_unique <= 80:
                vals = profile.top_values.get(col, [])
                if not vals:
                    vals = (
                        df[col].dropna().astype(str).value_counts().index[:80].tolist()
                    )
                values_str = f" | values: [{', '.join(str(v) for v in vals[:80])}]"
            elif kind == "categorical" and n_unique <= 200:
                vals = profile.top_values.get(col, [])[:20]
                values_str = f" | top-20 values: [{', '.join(str(v) for v in vals)}] … and {n_unique - 20} more"
            elif kind == "numeric":
                col_data = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(col_data):
                    values_str = (
                        f" | min={col_data.min():.2g}, "
                        f"max={col_data.max():.2g}, "
                        f"mean={col_data.mean():.2g}"
                    )
            elif kind == "datetime":
                dt_series = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(dt_series):
                    values_str = (
                        f" | range: {dt_series.min().date()} → {dt_series.max().date()}"
                    )

            missing_note = f", {missing_pct:.0f}% missing" if missing_pct > 5 else ""
            catalog_lines.append(
                f"  • {col}  [{kind}, {n_unique} unique{missing_note}]"
                f"{values_str}{ambiguous_note}"
            )

        column_catalog = "\n".join(catalog_lines)

        # ── Numeric statistics ───────────────────────────────────────────────
        num_cols = [c for c in profile.numeric_columns if c in df.columns]
        if num_cols:
            stats = df[num_cols].describe().round(2).to_string()
        else:
            stats = "No numeric columns."

        # ── Sample rows ──────────────────────────────────────────────────────
        sample_n = min(8, len(df))
        try:
            sample = df.head(sample_n).to_markdown(index=False)
        except Exception:
            sample = df.head(sample_n).to_string(index=False)

        # ── Derived metrics section ──────────────────────────────────────────
        derived_section = ""
        if profile.derived_metrics:
            lines = [
                "────────────────────────────────────────────────────────────────",
                "DERIVED METRICS (auto-computed — USE THESE IN CHARTS)",
                "────────────────────────────────────────────────────────────────",
            ]
            for dm in profile.derived_metrics:
                lines.append(f"  • {dm.name}  [{dm.unit}]: {dm.description}")
            derived_section = "\n".join(lines) + "\n\n"

        # ── Resource hint for the filter rule ───────────────────────────────
        # Pick the most likely "item/resource" column for the example in the prompt
        resource_hint = "resource"
        for candidate in ("resource", "item", "device", "equipment", "asset",
                          "product", "type", "category"):
            if candidate in all_cols:
                resource_hint = candidate
                break

        return _SYSTEM_PROMPT_TEMPLATE.format(
            file_hint=file_name,
            rows=len(df),
            n_cols=len(df.columns),
            sheet_section=sheet_section,
            column_catalog=column_catalog,
            stats=stats,
            sample_n=sample_n,
            sample=sample,
            derived_section=derived_section,
            resource_hint=resource_hint,
        )

    # ── Semantic pre-processing ──────────────────────────────────────────────

    @staticmethod
    def _annotate_query(
        user_prompt: str, df: pd.DataFrame, profile: DatasetProfile
    ) -> str:
        """
        Scan the user query for entity mentions.  When a term maps to specific
        column values in the real data, append a hidden analyst note to the
        prompt so the LLM knows exactly which column/values to use.

        Example
        -------
        Input : "give me the chart for charger duration"
        Output: "give me the chart for charger duration

        [ANALYST NOTE — use these mappings:
         • 'charger' → filter column 'resource' for values: Charger, USB Charger
         • 'duration' → use derived column 'duration_hours' (hours)]"
        """
        query_lower = user_prompt.lower()
        notes: List[str] = []

        # 1. Look for category-value matches (user mentions a specific item)
        for col in profile.categorical_columns:
            vals = profile.top_values.get(col, [])
            if not vals:
                vals = df[col].dropna().astype(str).unique().tolist()
            for val in vals:
                val_lower = val.lower()
                # Direct substring match: "charger" in query AND "charger" in val
                if len(val_lower) >= 3 and val_lower in query_lower:
                    notes.append(
                        f"• '{val_lower}' matches actual value '{val}' in column '{col}'"
                        f" — use filter_col='{col}', filter_val='{val}'"
                    )

        # 2. Check synonym dictionary for concept → column mapping
        for term, synonyms in _TERM_SYNONYMS.items():
            if term not in query_lower:
                continue
            # Check if any synonym matches a column name in the dataset
            for syn in synonyms:
                matched_cols = [
                    c for c in df.columns
                    if syn.replace("_", " ") in c.lower().replace("_", " ")
                ]
                for mc in matched_cols:
                    if mc in profile.numeric_columns or mc in profile.datetime_columns:
                        notes.append(
                            f"• '{term}' may refer to column '{mc}'"
                            f" [{str(df[mc].dtype)}]"
                        )

        # 3. Mention derived metrics that are relevant to the query
        for dm in profile.derived_metrics:
            dm_words = set(dm.name.replace("_", " ").split() + dm.unit.split())
            if any(w in query_lower for w in dm_words if len(w) > 3):
                notes.append(
                    f"• derived column '{dm.name}' ({dm.unit}) is available"
                    f" — {dm.description}"
                )

        if not notes:
            return user_prompt  # nothing to annotate

        annotation = (
            "\n\n[ANALYST CONTEXT — dataset mappings for this query:\n"
            + "\n".join(notes)
            + "\nUse these exact column names and values in your answer.]"
        )
        return user_prompt + annotation

    # ── Chart extractor ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_charts(
        raw: str, profile: DatasetProfile, df: pd.DataFrame
    ) -> List[ChartSpec]:
        """
        Parse <chart>{...}</chart> blocks from the LLM response into ChartSpec
        objects, with robust column resolution (case-insensitive + fuzzy).
        """
        charts: List[ChartSpec] = []
        all_cols: List[str] = list(df.columns)
        all_cols_lower = [c.lower() for c in all_cols]

        for match in re.finditer(r"<chart>(.*?)</chart>", raw, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

            chart_type = str(data.get("chart_type") or "bar").lower()
            x          = data.get("x") or data.get("x_col") or None
            y          = data.get("y") or data.get("y_col") or None
            agg        = str(data.get("agg") or "mean").lower()
            title      = str(data.get("title") or f"{chart_type.title()} chart")
            filter_col = data.get("filter_col") or None
            filter_val = data.get("filter_val") or None

            top_n_raw = data.get("top_n")
            top_n = int(top_n_raw) if top_n_raw is not None else 15

            # Normalise null-string sentinels
            for null_str in (None, "null", "None", ""):
                if x == null_str:
                    x = None
                if y == null_str:
                    y = None
                if filter_col == null_str:
                    filter_col = None
                if filter_val == null_str:
                    filter_val = None

            # Resolve column names
            x          = LLMChatEngine._resolve_col(x, all_cols, all_cols_lower)
            y          = LLMChatEngine._resolve_col(y, all_cols, all_cols_lower)
            filter_col = LLMChatEngine._resolve_col(filter_col, all_cols, all_cols_lower)

            if not x:
                # Heatmap doesn't need x
                if chart_type != "heatmap":
                    continue

            sort_by = "y_desc" if chart_type == "bar" else None

            charts.append(ChartSpec(
                title=title,
                chart_type=chart_type,
                x=x,
                y=y,
                agg=agg,
                top_n=top_n,
                sort_by=sort_by,
                filter_col=filter_col,
                filter_val=filter_val,
            ))

        return charts

    # ── Column resolver ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_col(
        name: Optional[str],
        all_cols: List[str],
        all_cols_lower: List[str],
    ) -> Optional[str]:
        """
        Resolve a column name the LLM produced to an actual column in the df.

        Resolution order:
          1. Exact match
          2. Case-insensitive exact match
          3. Underscore ↔ space normalised match
          4. Fuzzy best-match (SequenceMatcher ratio ≥ 0.80)
          5. None  (LLM hallucinated a column that doesn't exist)
        """
        if not name or name in (None, "null", "None", ""):
            return None

        # 1. Exact
        if name in all_cols:
            return name

        name_lower = name.lower()
        name_norm  = name_lower.replace(" ", "_").replace("-", "_")

        # 2. Case-insensitive
        for i, cl in enumerate(all_cols_lower):
            if cl == name_lower:
                return all_cols[i]

        # 3. Normalised underscore/space
        for i, cl in enumerate(all_cols_lower):
            cl_norm = cl.replace(" ", "_").replace("-", "_")
            if cl_norm == name_norm:
                return all_cols[i]

        # 4. Fuzzy match
        best_ratio, best_col = 0.0, None
        for i, cl in enumerate(all_cols_lower):
            ratio = SequenceMatcher(None, name_norm, cl.replace(" ", "_")).ratio()
            if ratio > best_ratio:
                best_ratio, best_col = ratio, all_cols[i]

        if best_ratio >= 0.80:
            return best_col

        return None  # truly unknown column

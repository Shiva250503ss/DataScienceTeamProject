from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.schemas import DatasetProfile, SemanticGroup

# Keyword library for rule-based grouping (no LLM needed).
# Key = category label, value = list of substrings to match (case-insensitive).
_CATEGORY_LIBRARY: Dict[str, List[str]] = {
    "Projector":   ["projector", "pico projector", "mini projector", "beamer"],
    "Camera":      ["camera", "canon", "nikon", "sony alpha", "gopro", "dslr", "webcam", "lens"],
    "Charger":     ["charger", "power adapter", "usb c", "usb-c", "magsafe", "charging cable"],
    "Phone":       ["iphone", "android", "pixel", "samsung", "phone", "smartphone"],
    "Mouse":       ["mouse", "wireless mouse", "bluetooth mouse", "trackpad"],
    "Keyboard":    ["keyboard", "mechanical keyboard", "numpad"],
    "Laptop":      ["laptop", "macbook", "thinkpad", "dell xps", "notebook", "chromebook"],
    "Audio":       ["mic", "microphone", "speaker", "headphone", "headset", "airpods", "earbuds"],
    "Storage":     ["ssd", "hard drive", "usb drive", "flash drive", "memory card", "sd card"],
    "Tablet":      ["ipad", "tablet", "surface pro"],
    "Monitor":     ["monitor", "display", "screen"],
    "Accessories": ["grip", "tripod", "stand", "mount", "cable", "case", "bag", "filter"],
}


def _rule_match_category(value: str) -> Optional[str]:
    """Return the first matching category label for a value, or None."""
    text = str(value).strip().lower()
    for category, keywords in _CATEGORY_LIBRARY.items():
        if any(kw in text for kw in keywords):
            return category
    return None


class SemanticGrouper:
    """
    Uses an LLM to group raw categorical values into broader semantic categories.

    Example:
        device column: "Canon EOS R5", "Nikon D3500", "USB-C Charger", "iPhone Charger"
        → creates device_category column: "Camera", "Camera", "Charger", "Charger"

    Returns (enriched_df, groups).
    If no LLM client is provided, returns the original df unchanged with an empty list.
    Falls back silently on any LLM or parsing failure.
    """

    def __init__(self, client: Any = None, model: Optional[str] = None):
        self.client = client
        self.model = model

    def _build_prompt(self, column: str, value_frequencies: Dict[str, int]) -> str:
        freq_json = json.dumps(value_frequencies, ensure_ascii=False, indent=2)
        return f"""You are a data analyst. Classify these values from column "{column}" into broader semantic categories.

Values and their counts:
{freq_json}

Return ONLY valid JSON in this exact format:
{{
  "derived_column": "snake_case_name_for_new_category_column",
  "mapping": {{
    "value1": "Category A",
    "value2": "Category A",
    "value3": "Category B"
  }},
  "confidence": 0.85,
  "reasoning": "brief explanation"
}}

Rules:
- Map every value listed above.
- Use 2 to 6 high-level categories. Use "Other" sparingly and only when values genuinely do not fit.
- derived_column must be a valid Python snake_case identifier (e.g. "device_category", "product_type").
- confidence is 0.0 to 1.0 — how meaningful and reliable is this grouping?
- If values cannot be meaningfully grouped (random IDs, zip codes, UUIDs, free-text), return:
  {{"derived_column": null, "mapping": {{}}, "confidence": 0.0, "reasoning": "not groupable"}}"""

    def _parse_and_validate(
        self,
        response_text: str,
        source_column: str,
        df: pd.DataFrame,
    ) -> Optional[SemanticGroup]:
        try:
            data = json.loads(response_text)
        except (json.JSONDecodeError, ValueError):
            return None

        derived_column = data.get("derived_column")
        mapping = data.get("mapping", {})
        confidence = float(data.get("confidence", 0.0))

        if not derived_column or not isinstance(mapping, dict):
            return None
        if confidence < 0.5:
            return None
        if len(set(mapping.values())) < 2:
            return None

        # Coverage: fraction of non-null rows whose value appears in the mapping
        series = df[source_column].dropna().astype(str)
        if len(series) == 0:
            return None
        coverage = series.isin(mapping.keys()).sum() / len(series)
        if coverage < 0.5:
            return None

        return SemanticGroup(
            source_column=source_column,
            derived_column=str(derived_column),
            mapping=mapping,
            coverage=float(coverage),
            confidence=confidence,
        )

    def _rule_based_run(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> Tuple[pd.DataFrame, List[SemanticGroup]]:
        """
        Rule-based grouping using _CATEGORY_LIBRARY — no LLM needed.
        Works on columns with 5–150 unique values where keywords match ≥50% of rows.
        """
        enriched = df.copy()
        groups: List[SemanticGroup] = []

        for col in profile.categorical_columns:
            # Skip text-heavy columns (free-form descriptions)
            if col in profile.text_columns:
                continue
            n_unique = profile.unique_summary.get(col, 0)
            if n_unique < 3 or n_unique > 150:
                continue

            series = enriched[col].dropna().astype(str)
            unique_values = series.unique().tolist()

            mapping: Dict[str, str] = {}
            for val in unique_values:
                cat = _rule_match_category(val)
                if cat:
                    mapping[val] = cat

            # Need ≥2 distinct categories and ≥50% row coverage
            if len(set(mapping.values())) < 2:
                continue
            coverage = series.isin(mapping.keys()).sum() / max(len(series), 1)
            if coverage < 0.5:
                continue

            derived_col = f"{col}_category"
            enriched[derived_col] = enriched[col].astype(str).map(mapping)
            groups.append(SemanticGroup(
                source_column=col,
                derived_column=derived_col,
                mapping=mapping,
                coverage=float(coverage),
                confidence=0.75,
            ))

        return enriched, groups

    def run(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> Tuple[pd.DataFrame, List[SemanticGroup]]:
        """
        Analyse all categorical columns. For columns where the LLM finds a
        meaningful grouping, add a derived column to the returned dataframe.
        Falls back to rule-based keyword grouping when no LLM client is configured.
        """
        enriched = df.copy()
        groups: List[SemanticGroup] = []

        if self.client is None:
            return self._rule_based_run(df, profile)

        for col in profile.categorical_columns:
            freqs = profile.value_frequencies.get(col, {})
            if len(freqs) < 3:          # too few values to be worth grouping
                continue
            if len(freqs) == len(set(freqs.keys())):
                pass                    # normal dict — proceed

            try:
                resp = self.client.generate(self._build_prompt(col, freqs), self.model)
                group = self._parse_and_validate(resp, col, df)
                if group:
                    # Add derived column; unmapped values become NaN
                    enriched[group.derived_column] = (
                        enriched[col].astype(str).map(group.mapping)
                    )
                    groups.append(group)
            except Exception:
                continue   # silent fallback — original data unaffected

        return enriched, groups

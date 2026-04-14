from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class LLMColumnResult:
    column: str
    semantic_type: str
    importance_score: float   # 0 to 1
    is_identifier: bool
    reasoning: str


class LLMColumnIntelligence:
    """
    Provider-agnostic LLM intelligence layer.
    Pass any client that implements client.generate(prompt, model) -> str.
    If no client is configured, falls back to rule-based keyword inference.
    """

    def __init__(self, client: Any = None, model: Optional[str] = None):
        self.client = client
        self.model = model

    @staticmethod
    def _build_column_payload(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        series = df[column]
        sample_values = series.dropna().astype(str).head(8).tolist()
        return {
            "column_name": column,
            "dtype": str(series.dtype),
            "sample_values": sample_values,
            "nunique": int(series.nunique(dropna=True)),
            "missing_ratio": float(series.isna().mean()),
        }

    @staticmethod
    def _fallback_rule_based(column: str) -> LLMColumnResult:
        col = column.lower()

        if any(k in col for k in ["id", "uuid", "key", "customer_id", "order_id"]):
            return LLMColumnResult(
                column=column,
                semantic_type="identifier",
                importance_score=0.0,
                is_identifier=True,
                reasoning="Column name looks like an identifier.",
            )

        if any(k in col for k in ["sales", "revenue", "amount", "gmv"]):
            return LLMColumnResult(
                column=column,
                semantic_type="revenue",
                importance_score=0.95,
                is_identifier=False,
                reasoning="Column name suggests revenue-like metric.",
            )

        if "profit" in col:
            return LLMColumnResult(
                column=column,
                semantic_type="profit",
                importance_score=0.9,
                is_identifier=False,
                reasoning="Column name suggests profit metric.",
            )

        if any(k in col for k in ["cost", "expense", "cogs", "spend"]):
            return LLMColumnResult(
                column=column,
                semantic_type="cost",
                importance_score=0.8,
                is_identifier=False,
                reasoning="Column name suggests cost metric.",
            )

        if any(k in col for k in ["qty", "quantity", "units", "count", "volume"]):
            return LLMColumnResult(
                column=column,
                semantic_type="quantity",
                importance_score=0.75,
                is_identifier=False,
                reasoning="Column name suggests quantity/count metric.",
            )

        if any(k in col for k in ["date", "time", "created", "updated", "timestamp"]):
            return LLMColumnResult(
                column=column,
                semantic_type="datetime",
                importance_score=0.7,
                is_identifier=False,
                reasoning="Column name suggests datetime field.",
            )

        return LLMColumnResult(
            column=column,
            semantic_type="unknown",
            importance_score=0.4,
            is_identifier=False,
            reasoning="No strong semantic signal found.",
        )

    def analyze_column(self, df: pd.DataFrame, column: str) -> LLMColumnResult:
        """Score a single column. Falls back to rule-based if no LLM client."""
        if self.client is None:
            return self._fallback_rule_based(column)

        payload = self._build_column_payload(df, column)

        prompt = f"""You are helping an AI dashboard generator understand dataset columns.

Given this column metadata:
{json.dumps(payload, indent=2)}

Return ONLY valid JSON with this schema:
{{
  "semantic_type": "one of [revenue, profit, cost, quantity, price, datetime, category, identifier, unknown]",
  "importance_score": "number from 0 to 1",
  "is_identifier": "true or false",
  "reasoning": "short explanation"
}}

Rules:
- Mark identifier=true for columns that are IDs, keys, unique codes, or row identifiers.
- Give higher importance to columns that are likely core business metrics.
- Be conservative. Do not guess wildly."""

        try:
            response_text = self.client.generate(prompt=prompt, model=self.model)
            data = json.loads(response_text)

            return LLMColumnResult(
                column=column,
                semantic_type=str(data.get("semantic_type", "unknown")),
                importance_score=float(data.get("importance_score", 0.4)),
                is_identifier=bool(data.get("is_identifier", False)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception:
            return self._fallback_rule_based(column)

    def analyze_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Dict[str, LLMColumnResult]:
        results: Dict[str, LLMColumnResult] = {}
        for col in columns:
            if col in df.columns:
                results[col] = self.analyze_column(df, col)
        return results

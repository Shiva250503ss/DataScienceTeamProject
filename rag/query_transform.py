# rag/query_transform.py

"""
Module 3 — Query transformation for the Data Analyzer's chat mode.

THE PROBLEM
  Users of the chat interface ask things like:
      "what about the west region?"          (multi-turn — depends on history)
      "why is it dropping?"                  (vague — 'it'? dropping what?)
      "show revenue but per store this time" (delta on a previous request)
  Feeding these raw into retrieval OR into the chart-spec LLM gives garbage:
  BM25 has no terms to match, embeddings embed the vagueness.

THE FIX — rewrite before retrieve
  A small LLM call (local Mistral) rewrites the question into a fully
  self-contained form using the chat history and the dataset's actual
  column names. One rewrite, used twice:
      1. as the retrieval query (hybrid_search + reranker)
      2. as the clarified request handed to the chart-spec generator

  We also generate 2 QUERY EXPANSIONS (paraphrases emphasizing different
  terms) — cheap recall boost for the BM25 arm: the union of hits from all
  variants feeds the reranker, which restores precision.

Falls back to the original query on any LLM failure — the chat never breaks
because of this module.
"""

import json
import os
import re
from typing import Dict, List, Optional

import requests

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")

REWRITE_PROMPT = """You clean up analytics questions before they go to a retrieval system.

DATASET COLUMNS: {columns}

CONVERSATION SO FAR (may be empty):
{history}

USER'S LATEST QUESTION: "{query}"

Rewrite the question to be fully self-contained:
- resolve pronouns ("it", "that", "those") using the conversation
- replace vague words with the specific dataset column names they refer to
- keep it ONE sentence
Also produce 2 short paraphrases that emphasize different key terms.

Answer with ONLY this JSON:
{{"rewritten": "...", "expansions": ["...", "..."]}}"""


def _ollama_json(prompt: str) -> Optional[Dict]:
    """One JSON-constrained generation via local Ollama. None on failure."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "format": "json",
                  "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 200}},
            timeout=60,
        )
        r.raise_for_status()
        return json.loads(r.json()["response"])
    except Exception:
        return None


class QueryTransformer:
    """Rewrites vague/multi-turn questions into retrieval-ready queries."""

    def transform(self, query: str,
                  chat_history: Optional[List[Dict]] = None,
                  columns: Optional[List[str]] = None) -> Dict:
        """
        Args:
            query:        the user's latest message
            chat_history: [{"role": "user"/"assistant", "content": ...}, ...]
            columns:      dataset column names (grounds the rewrite in real
                          columns instead of invented ones)

        Returns:
            {"original": ..., "rewritten": ..., "expansions": [...],
             "was_rewritten": bool}
        """
        result = {"original": query, "rewritten": query,
                  "expansions": [], "was_rewritten": False}

        # Cheap short-circuit: long, specific, first-turn questions rarely
        # benefit from a rewrite — skip the LLM call entirely.
        is_first_turn = not chat_history
        looks_specific = len(query.split()) >= 8 and not re.search(
            r"\b(it|that|those|this one|them|why is|what about)\b", query.lower())
        if is_first_turn and looks_specific:
            return result

        history_text = "\n".join(
            f"{m['role']}: {m['content'][:200]}"
            for m in (chat_history or [])[-6:]   # last 3 exchanges are enough
        ) or "(no prior messages)"

        parsed = _ollama_json(REWRITE_PROMPT.format(
            columns=", ".join(columns or [])[:600] or "(unknown)",
            history=history_text,
            query=query,
        ))
        if parsed and parsed.get("rewritten"):
            result["rewritten"] = str(parsed["rewritten"]).strip()
            result["expansions"] = [str(e).strip()
                                    for e in parsed.get("expansions", [])[:2]]
            result["was_rewritten"] = result["rewritten"].lower() != query.lower()
        return result

    def retrieval_queries(self, query: str,
                          chat_history: Optional[List[Dict]] = None,
                          columns: Optional[List[str]] = None) -> List[str]:
        """Rewritten query + expansions, deduped — feed each to hybrid search."""
        t = self.transform(query, chat_history, columns)
        queries = [t["rewritten"]] + t["expansions"]
        seen, out = set(), []
        for q in queries:
            if q.lower() not in seen and q.strip():
                seen.add(q.lower())
                out.append(q)
        return out


if __name__ == "__main__":
    qt = QueryTransformer()
    demo_history = [
        {"role": "user", "content": "show me revenue by region"},
        {"role": "assistant", "content": "Here is revenue by region — West leads with $1.2M."},
    ]
    print(qt.transform("why is it dropping?", demo_history,
                       ["region", "revenue", "month", "store_id"]))

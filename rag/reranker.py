# rag/reranker.py

"""
Module 2 — Cross-encoder reranking on top of hybrid retrieval results.

WHY RERANK AT ALL?
  Bi-encoders (the MiniLM embedder) compress query and document into vectors
  INDEPENDENTLY, then compare. Fast, but the model never sees the query and
  document together, so it misses fine-grained relevance ("high income
  REDUCES churn risk" vs "high income INCREASES churn risk" embed almost
  identically). A cross-encoder feeds the concatenated (query, document)
  pair through the transformer jointly and scores true relevance.

WHY NOT CROSS-ENCODE EVERYTHING?
  Cost: the cross-encoder is O(candidates) full transformer passes per
  query. The standard two-stage design is:
      stage 1 (cheap, wide):  hybrid search  -> ~20 candidates
      stage 2 (precise, narrow): cross-encoder -> best 3-5
  This is exactly the retrieve-then-rerank pattern used in production RAG.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (default, configurable) —
trained on MS MARCO passage ranking, 22M params, runs fine on CPU. Local,
no API key.
"""

from typing import Dict, List, Optional

_cross_encoder = None
_load_failed = False


def _get_cross_encoder():
    """Lazy singleton, same pattern as rag.embeddings."""
    global _cross_encoder, _load_failed
    if _cross_encoder is None and not _load_failed:
        try:
            from sentence_transformers import CrossEncoder
            from utils.config import config
            _cross_encoder = CrossEncoder(config.RAG_RERANKER_MODEL, max_length=512)
        except Exception as e:
            print(f"[rag.reranker] cross-encoder unavailable ({e}) — "
                  f"falling back to hybrid RRF order")
            _load_failed = True
    return _cross_encoder


class Reranker:
    """Rerank hybrid candidates with a local cross-encoder."""

    def rerank(self, query: str, candidates: List[Dict],
               top_k: int = 4) -> List[Dict]:
        """
        Score each (query, candidate_text) pair jointly; return the top_k
        candidates sorted by cross-encoder score (added as 'rerank_score').

        Falls back to the incoming (RRF) order when the model is unavailable,
        so callers never need a special code path.
        """
        if not candidates:
            return []
        model = _get_cross_encoder()
        if model is None:
            return candidates[:top_k]

        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"],
                      reverse=True)[:top_k]


def retrieve_and_rerank(query: str, top_k: int = 4,
                        fetch_k: int = 20) -> List[Dict]:
    """
    Convenience one-liner used by the agents:
        hybrid search (fetch_k wide) -> cross-encoder rerank -> top_k.
    Returns [] on any failure so agent code stays clean.
    """
    try:
        from rag.hybrid_search import HybridSearcher
        candidates = HybridSearcher().search(query, top_k=fetch_k, fetch_k=fetch_k)
        return Reranker().rerank(query, candidates, top_k=top_k)
    except Exception as e:
        print(f"[rag] retrieve_and_rerank failed ({e}) — continuing without context")
        return []


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "which features drove churn"
    for r in retrieve_and_rerank(q):
        print(f"  [{r.get('rerank_score', 0):.3f}] ({r['doc_type']}) {r['text'][:100]}...")

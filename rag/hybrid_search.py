# rag/hybrid_search.py

"""
Module 1 — Hybrid retrieval: dense (Qdrant) + sparse (BM25) merged with
Reciprocal Rank Fusion.

WHY HYBRID?
  Dense and sparse retrieval fail in OPPOSITE ways:
    - Dense (embeddings) understands paraphrase ("customers leaving" matches
      "churn") but is weak on exact rare tokens — column names like
      'debt_to_income_ratio' or dataset names embed poorly.
    - BM25 nails exact terms/identifiers but has zero semantic understanding.
  Our corpus is full of BOTH: natural-language explanations AND snake_case
  column names. Hybrid covers both failure modes.

WHY RRF (Reciprocal Rank Fusion)?
  Dense scores are cosine similarities (~0.2-0.9); BM25 scores are unbounded
  (0-30+). Normalizing incompatible score scales is fragile. RRF sidesteps
  the problem by fusing on RANKS only:

      RRF(doc) = sum over retrievers of  1 / (k + rank_in_that_retriever)

  with k=60 (the value from the original Cormack et al. paper — it dampens
  the head so one retriever can't dominate). A doc found by BOTH retrievers
  gets two contributions and floats to the top.

Usage:
    searcher = HybridSearcher()
    results = searcher.search("what drove churn in the telco dataset", top_k=8)
    # each result: {"id","text","doc_type","dataset_name","rrf_score", ...}
"""

import re
from typing import Dict, List, Optional

from rag.embeddings import embed_one
from rag.indexer import RagIndexer

RRF_K = 60  # standard damping constant from the RRF paper


def _tokenize(text: str) -> List[str]:
    """
    BM25 tokenizer: lowercase word chars, with snake_case identifiers split
    into their parts AND kept whole — so 'debt_to_income_ratio' matches both
    the exact identifier and a query that says 'income ratio'.
    """
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    out = []
    for t in tokens:
        out.append(t)
        if "_" in t:
            out.extend(p for p in t.split("_") if len(p) > 2)
    return out


class HybridSearcher:
    """Dense + sparse retrieval over the RAG corpus, fused with RRF."""

    def __init__(self, indexer: Optional[RagIndexer] = None):
        self.indexer = indexer or RagIndexer()
        self._bm25 = None
        self._bm25_corpus: List[Dict] = []
        self._corpus_size_at_build = -1

    # ── Sparse arm ────────────────────────────────────────────────────────

    def _ensure_bm25(self):
        """(Re)build the BM25 index if the corpus changed since last build."""
        corpus = self.indexer.load_corpus()
        if len(corpus) == self._corpus_size_at_build:
            return
        self._bm25_corpus = corpus
        self._corpus_size_at_build = len(corpus)
        if not corpus:
            self._bm25 = None
            return
        try:
            # BM25Plus, not BM25Okapi: Okapi's IDF is ln((N-df+0.5)/(df+0.5)),
            # which is EXACTLY 0 for a term in half the corpus — on a small
            # knowledge base (first few pipeline runs) every match scores 0
            # and sparse retrieval silently returns nothing. BM25Plus adds a
            # lower bound (delta) so any term match contributes positively.
            from rank_bm25 import BM25Plus
            self._bm25 = BM25Plus([_tokenize(d["text"]) for d in corpus])
        except ImportError:
            print("[rag.hybrid] rank_bm25 not installed — sparse arm disabled "
                  "(pip install rank-bm25)")
            self._bm25 = None

    def _sparse_search(self, query: str, top_k: int) -> List[Dict]:
        self._ensure_bm25()
        if self._bm25 is None or not self._bm25_corpus:
            return []
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # BM25Plus floors every score above 0 (even zero-match docs), so a
        # score filter can't tell hits from non-hits. Require an explicit
        # token overlap instead — a doc must share at least one query token.
        q_set = set(q_tokens)
        results = []
        for i in ranked:
            if len(results) >= top_k:
                break
            if q_set & set(_tokenize(self._bm25_corpus[i]["text"])):
                results.append({**self._bm25_corpus[i],
                                "bm25_score": float(scores[i])})
        return results

    # ── Dense arm ─────────────────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int) -> List[Dict]:
        client = self.indexer._qdrant()
        if client is None:
            return []
        vector = embed_one(query)
        if vector is None:
            return []
        try:
            hits = client.search(
                collection_name=self.indexer.collection,
                query_vector=vector,
                limit=top_k,
            )
            return [{**h.payload, "dense_score": float(h.score)} for h in hits]
        except Exception as e:
            print(f"[rag.hybrid] Qdrant search failed ({e}) — dense arm skipped")
            return []

    # ── Fusion ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 8,
               fetch_k: int = 20) -> List[Dict]:
        """
        Hybrid search: retrieve fetch_k candidates from each arm, fuse with
        RRF, return the top_k. fetch_k > top_k on purpose — RRF needs deep
        candidate lists to surface docs that rank mid-tier in BOTH arms.
        """
        dense = self._dense_search(query, fetch_k)
        sparse = self._sparse_search(query, fetch_k)

        fused: Dict[str, Dict] = {}
        for rank, doc in enumerate(dense):
            entry = fused.setdefault(doc["id"], {**doc, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
        for rank, doc in enumerate(sparse):
            entry = fused.setdefault(doc["id"], {**doc, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            entry.setdefault("bm25_score", doc.get("bm25_score"))

        results = sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    # Quick manual test: python -m rag.hybrid_search "your query"
    import sys
    q = " ".join(sys.argv[1:]) or "which features drove churn"
    searcher = HybridSearcher()
    print(f"Query: {q}\nCorpus: {searcher.indexer.corpus_stats()}")
    for r in searcher.search(q):
        print(f"  [{r['rrf_score']:.4f}] ({r['doc_type']}/{r['dataset_name']}) "
              f"{r['text'][:100]}...")

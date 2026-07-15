# rag/retrieval_eval.py

"""
Retrieval quality evaluation — recall@k, nDCG@k, MRR over a labeled query set.

WHAT IT MEASURES
  Four retrieval configurations, so each pipeline stage's contribution is
  visible in the numbers instead of asserted:
      1. bm25    — sparse only
      2. dense   — vector only (Qdrant)
      3. hybrid  — BM25 + dense fused with RRF
      4. hybrid+rerank — cross-encoder on top (the production config)

LABELED SET
  rag/eval_queries.json — [{"query": ..., "relevant_doc_ids": [...]}, ...]
  built from REAL indexed documents (see --list to enumerate the corpus with
  ids). Relevance is binary; nDCG uses gain 1 for relevant docs.

METRICS (standard IR definitions)
  recall@k = |relevant ∩ top-k| / |relevant|
  nDCG@k   = DCG@k / IDCG@k,  DCG = Σ rel_i / log2(i+1)  (i is 1-based rank)
  MRR      = 1 / rank of the first relevant result (0 if none retrieved)

Usage:
    python -m rag.retrieval_eval --list          # show corpus ids for labeling
    python -m rag.retrieval_eval                 # evaluate all 4 configs
    python -m rag.retrieval_eval --k 5 --out rag/eval_results.md
"""

import argparse
import json
import math
import os
from typing import Dict, List

from rag.hybrid_search import HybridSearcher
from rag.indexer import RAG_STORE_DIR, RagIndexer
from rag.reranker import Reranker

EVAL_QUERIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "eval_queries.json")


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(retrieved_ids: List[str], relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2)
              for i, doc_id in enumerate(retrieved_ids[:k])
              if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: List[str], relevant: set) -> float:
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Retrieval configurations ──────────────────────────────────────────────────

def run_config(name: str, searcher: HybridSearcher, reranker: Reranker,
               query: str, k: int) -> List[str]:
    """Return ranked doc ids for one configuration."""
    fetch = max(20, k * 4)
    if name == "bm25":
        docs = searcher._sparse_search(query, fetch)
    elif name == "dense":
        docs = searcher._dense_search(query, fetch)
    elif name == "hybrid":
        docs = searcher.search(query, top_k=fetch, fetch_k=fetch)
    elif name == "hybrid+rerank":
        candidates = searcher.search(query, top_k=fetch, fetch_k=fetch)
        docs = reranker.rerank(query, candidates, top_k=k)
    else:
        raise ValueError(name)
    return [d["id"] for d in docs]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--queries", type=str, default=EVAL_QUERIES_FILE)
    parser.add_argument("--out", type=str,
                        default=os.path.join(RAG_STORE_DIR, "retrieval_eval_results.md"))
    parser.add_argument("--list", action="store_true",
                        help="List indexed docs with ids (for building labels)")
    args = parser.parse_args()

    corpus = RagIndexer.load_corpus()
    if args.list:
        print(f"Corpus: {len(corpus)} documents\n")
        for d in corpus:
            print(f"{d['id']}  [{d['doc_type']}/{d['dataset_name']}] "
                  f"{d['text'][:110]}...")
        return

    if not os.path.exists(args.queries):
        raise SystemExit(
            f"{args.queries} not found. Build the labeled set first:\n"
            f"  1. python -m rag.retrieval_eval --list   (see doc ids)\n"
            f"  2. write eval_queries.json: "
            f'[{{"query": "...", "relevant_doc_ids": ["..."]}}]')

    with open(args.queries, "r", encoding="utf-8") as f:
        labeled = json.load(f)

    corpus_ids = {d["id"] for d in corpus}
    # Guard against stale labels pointing at re-indexed/removed docs
    for entry in labeled:
        missing = [i for i in entry["relevant_doc_ids"] if i not in corpus_ids]
        if missing:
            raise SystemExit(f"Label references unknown doc ids {missing} for "
                             f"query '{entry['query']}' — rebuild labels "
                             f"against the current corpus (--list).")

    print(f"Corpus: {len(corpus)} docs | labeled queries: {len(labeled)} | k={args.k}")

    searcher = HybridSearcher()
    reranker = Reranker()
    configs = ["bm25", "dense", "hybrid", "hybrid+rerank"]

    rows = []
    per_query_log = []
    for config_name in configs:
        recalls, ndcgs, mrrs = [], [], []
        for entry in labeled:
            relevant = set(entry["relevant_doc_ids"])
            try:
                retrieved = run_config(config_name, searcher, reranker,
                                       entry["query"], args.k)
            except Exception as e:
                print(f"  ! {config_name} failed on '{entry['query']}': {e}")
                retrieved = []
            recalls.append(recall_at_k(retrieved, relevant, args.k))
            ndcgs.append(ndcg_at_k(retrieved, relevant, args.k))
            mrrs.append(mrr(retrieved, relevant))
            per_query_log.append({
                "config": config_name, "query": entry["query"],
                "recall": recalls[-1], "ndcg": ndcgs[-1],
                "retrieved": retrieved[:args.k],
            })
        n = max(len(labeled), 1)
        rows.append({
            "config": config_name,
            f"recall@{args.k}": round(sum(recalls) / n, 4),
            f"ndcg@{args.k}": round(sum(ndcgs) / n, 4),
            "mrr": round(sum(mrrs) / n, 4),
        })
        print(f"  {config_name:15s} recall@{args.k}={rows[-1][f'recall@{args.k}']:.4f} "
              f"ndcg@{args.k}={rows[-1][f'ndcg@{args.k}']:.4f} "
              f"mrr={rows[-1]['mrr']:.4f}")

    # ── Markdown report ───────────────────────────────────────────────────
    md = ["# Retrieval Evaluation Results", "",
          f"Corpus: {len(corpus)} real indexed documents | "
          f"{len(labeled)} labeled queries | k={args.k}", "",
          f"| Config | recall@{args.k} | nDCG@{args.k} | MRR |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r[f'recall@{args.k}']} | "
                  f"{r[f'ndcg@{args.k}']} | {r['mrr']} |")
    md += ["", "## Per-query detail", "```json",
           json.dumps(per_query_log, indent=1), "```"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

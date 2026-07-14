# rag/ — Retrieval-Augmented Generation optimization pipeline.
#
# Finally puts the Qdrant container (deployed since day one, previously
# unused) to work. All models run locally — no API keys:
#
#   embeddings.py      sentence-transformers MiniLM embedder (384-dim, local)
#   indexer.py         document store: Qdrant (dense) + JSON mirror (for BM25)
#   hybrid_search.py   dense + BM25 sparse retrieval merged with RRF
#   reranker.py        cross-encoder reranking on top of hybrid results
#   query_transform.py LLM query rewriting for vague/multi-turn chat questions
#   graph_retrieval.py lightweight knowledge graph over columns/features/runs
#                      for multi-hop questions
#
# Wiring into the platform:
#   agents/explainer.py     — retrieves similar past explanations as few-shot
#                             context; indexes each run's narratives afterwards
#   agents/data_analyzer.py — rewrites chat queries before retrieval; indexes
#                             discovered insights; KG answers multi-hop questions
#
# Everything degrades gracefully: if Qdrant or sentence-transformers are
# unavailable, retrieval falls back to BM25-only; if that fails too, agents
# simply run without retrieval context (exactly as before this module existed).

from rag.hybrid_search import HybridSearcher  # noqa: F401
from rag.reranker import Reranker             # noqa: F401
from rag.indexer import RagIndexer            # noqa: F401

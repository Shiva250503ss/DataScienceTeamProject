# rag/embeddings.py

"""
Local text embeddings for the RAG pipeline.

Model: sentence-transformers/all-MiniLM-L6-v2 (default, configurable via
RAG_EMBEDDING_MODEL in .env).

Why this model:
  - 384 dimensions, 22M params — embeds thousands of docs/second on CPU,
    so retrieval adds no perceptible latency to the pipeline
  - trained specifically for semantic similarity (unlike raw LLM embeddings)
  - fully local: downloads once from HF hub, then works offline — consistent
    with the platform's "no API keys" constraint

The embedder is a lazy singleton: the model loads on first use, not at
import time, so `import rag` never slows down (or breaks) the main pipeline.
"""

from typing import List, Optional

_model = None
_load_failed = False


def get_embedder():
    """Return the shared SentenceTransformer, or None if unavailable."""
    global _model, _load_failed
    if _model is None and not _load_failed:
        try:
            from sentence_transformers import SentenceTransformer
            from utils.config import config
            _model = SentenceTransformer(config.RAG_EMBEDDING_MODEL)
        except Exception as e:
            print(f"[rag.embeddings] sentence-transformers unavailable ({e}) "
                  f"— dense retrieval disabled, BM25-only mode")
            _load_failed = True
    return _model


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed a batch of texts. Returns None when the embedder is unavailable."""
    model = get_embedder()
    if model is None:
        return None
    # normalize_embeddings=True -> cosine similarity == dot product, which is
    # what the Qdrant collection is configured for (Distance.COSINE)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> Optional[List[float]]:
    vectors = embed_texts([text])
    return vectors[0] if vectors else None


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size; must match Qdrant collection

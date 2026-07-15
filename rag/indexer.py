# rag/indexer.py

"""
Document ingestion for the RAG pipeline — writes to BOTH stores:

  1. Qdrant   — dense vectors for semantic search (the container from
                docker-compose that was deployed-but-unused until now)
  2. JSON mirror (rag_store/documents.json) — full texts for BM25 sparse
                retrieval, which needs the raw corpus in memory

Why a mirror instead of scrolling Qdrant for BM25?
  BM25 must tokenize the ENTIRE corpus to compute IDF. Scrolling thousands
  of payloads out of Qdrant on every query would defeat the point; a local
  JSON file (a few MB) loads in milliseconds and keeps the two stores in
  sync because ALL writes go through this one class.

Document types indexed by the platform:
  - "explanation"      — Explainer LLM narratives (with their SHAP context)
  - "insight"          — Data Analyzer discovered insights
  - "dataset_profile"  — dataset summary facts (columns, target, quality)

Every document carries: id, text, doc_type, dataset_name, created_at, metadata.
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

from rag.embeddings import EMBEDDING_DIM, embed_texts

# Local corpus mirror lives next to the repo root so it survives reruns
RAG_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "rag_store")
DOCUMENTS_FILE = os.path.join(RAG_STORE_DIR, "documents.json")


class RagIndexer:
    """Single write-path for the RAG knowledge base (Qdrant + JSON mirror)."""

    def __init__(self):
        from utils.config import config
        self.collection = config.RAG_COLLECTION
        self.qdrant_url = config.QDRANT_URL
        self._client = None
        self._qdrant_failed = False

    # ── Qdrant client (lazy, optional) ────────────────────────────────────

    def _qdrant(self):
        """
        Lazy Qdrant client. Preference order:
          1. Qdrant SERVER at QDRANT_URL (docker-compose / production)
          2. Qdrant EMBEDDED LOCAL MODE (qdrant-client's on-disk storage at
             rag_store/qdrant_local) — a real, persistent vector index that
             needs no server process. Keeps the dense arm working on dev
             machines without Docker.
        Returns None only if the qdrant-client library itself is missing.
        """
        if self._client is None and not self._qdrant_failed:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
                try:
                    client = QdrantClient(url=self.qdrant_url, timeout=5)
                    client.get_collections()  # probe: raises if no server
                except Exception:
                    local_path = os.path.join(RAG_STORE_DIR, "qdrant_local")
                    os.makedirs(RAG_STORE_DIR, exist_ok=True)
                    client = QdrantClient(path=local_path)
                    print(f"[rag.indexer] Qdrant server unreachable at "
                          f"{self.qdrant_url} — using embedded local mode "
                          f"({local_path})")
                # Create the collection on first contact (idempotent check)
                existing = [c.name for c in client.get_collections().collections]
                if self.collection not in existing:
                    client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=EMBEDDING_DIM,
                                                    distance=Distance.COSINE),
                    )
                self._client = client
            except Exception as e:
                print(f"[rag.indexer] qdrant-client unavailable ({e}) "
                      f"— dense index disabled, BM25 mirror still active")
                self._qdrant_failed = True
        return self._client

    # ── JSON mirror ───────────────────────────────────────────────────────

    @staticmethod
    def load_corpus() -> List[Dict]:
        """Load the full document corpus (used by BM25 and graph building)."""
        if not os.path.exists(DOCUMENTS_FILE):
            return []
        try:
            with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @staticmethod
    def _save_corpus(docs: List[Dict]):
        os.makedirs(RAG_STORE_DIR, exist_ok=True)
        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=1)

    # ── Public API ────────────────────────────────────────────────────────

    def index_documents(self, docs: List[Dict]) -> int:
        """
        Index a batch of documents into both stores.

        Each doc needs: {"text": str, "doc_type": str, "dataset_name": str,
                         "metadata": dict (optional)}
        The id is a content hash, so re-indexing the same run is idempotent
        (no duplicate documents from repeated pipeline runs).

        Returns the number of NEW documents added.
        """
        corpus = self.load_corpus()
        known_ids = {d["id"] for d in corpus}

        new_docs = []
        for doc in docs:
            text = (doc.get("text") or "").strip()
            if len(text) < 30:          # skip trivially short fragments
                continue
            doc_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            if doc_id in known_ids:
                continue
            known_ids.add(doc_id)
            new_docs.append({
                "id": doc_id,
                "text": text,
                "doc_type": doc.get("doc_type", "unknown"),
                "dataset_name": doc.get("dataset_name", "unknown"),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": doc.get("metadata", {}),
            })

        if not new_docs:
            return 0

        # 1. JSON mirror (always works — no external service)
        corpus.extend(new_docs)
        self._save_corpus(corpus)

        # 2. Qdrant dense vectors (best effort)
        client = self._qdrant()
        if client is not None:
            vectors = embed_texts([d["text"] for d in new_docs])
            if vectors is not None:
                from qdrant_client.models import PointStruct
                points = [
                    PointStruct(
                        # Qdrant point ids must be uint or UUID — use the
                        # first 16 hex chars of the md5 as an int
                        id=int(d["id"][:16], 16),
                        vector=vec,
                        payload={k: d[k] for k in
                                 ("id", "text", "doc_type", "dataset_name",
                                  "created_at", "metadata")},
                    )
                    for d, vec in zip(new_docs, vectors)
                ]
                try:
                    client.upsert(collection_name=self.collection, points=points)
                except Exception as e:
                    print(f"[rag.indexer] Qdrant upsert failed ({e}) — "
                          f"documents remain searchable via BM25")

        return len(new_docs)

    def corpus_stats(self) -> Dict:
        corpus = self.load_corpus()
        by_type: Dict[str, int] = {}
        for d in corpus:
            by_type[d["doc_type"]] = by_type.get(d["doc_type"], 0) + 1
        return {"total": len(corpus), "by_type": by_type}

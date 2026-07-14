# rag/graph_retrieval.py

"""
Module 4 — Lightweight knowledge graph for multi-hop questions.

WHY A GRAPH WHEN WE ALREADY HAVE VECTOR SEARCH?
  Vector/BM25 retrieval finds documents SIMILAR to the question. It cannot
  answer questions whose evidence is SPREAD ACROSS runs, e.g.:
      "how does monthly_charges relate to what drove churn in the telco data?"
  No single document contains that answer — but a graph walk does:
      monthly_charges --[predicts, shap=0.42]--> churn (telco run)
      monthly_charges --[correlates_with, r=0.87]--> total_charges
  Two hops, two facts from different pipeline stages, one answer.

GRAPH SCHEMA (deliberately tiny — no graph DB, just JSON + adjacency dicts):
  Nodes: dataset | column | target
  Edges (all carry a 'fact' sentence used verbatim as retrieval output):
    column --belongs_to--> dataset
    column --predicts--> target        (from SHAP importance, weight = |SHAP|)
    column --correlates_with--> column (from the feature agent's VIF/target
                                        correlation analysis, weight = |r| or VIF)

  Facts are added by the agents after each pipeline run via record_run(),
  and persisted to rag_store/knowledge_graph.json.

QUERYING
  1. Fuzzy-match entities mentioned in the question to graph nodes
  2. BFS out to `max_hops` (default 2)
  3. Return the traversed edges' fact sentences, strongest weights first
  These facts are appended to the agent's LLM context alongside the
  hybrid+rerank documents.
"""

import json
import os
import re
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from rag.indexer import RAG_STORE_DIR

GRAPH_FILE = os.path.join(RAG_STORE_DIR, "knowledge_graph.json")


class KnowledgeGraph:
    """JSON-backed multi-hop fact graph over datasets, columns, and targets."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}   # node_id -> {"type": ..., "name": ...}
        self.edges: List[Dict] = []        # {"src","dst","relation","weight","fact"}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(GRAPH_FILE):
            try:
                with open(GRAPH_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", [])
            except Exception:
                pass

    def save(self):
        os.makedirs(RAG_STORE_DIR, exist_ok=True)
        with open(GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f,
                      ensure_ascii=False, indent=1)

    # ── Building ──────────────────────────────────────────────────────────

    @staticmethod
    def _node_id(node_type: str, name: str) -> str:
        return f"{node_type}:{name.strip().lower()}"

    def _add_node(self, node_type: str, name: str) -> str:
        nid = self._node_id(node_type, name)
        self.nodes.setdefault(nid, {"type": node_type, "name": name})
        return nid

    def _add_edge(self, src: str, dst: str, relation: str,
                  weight: float, fact: str):
        # Dedupe on (src, dst, relation) — keep the strongest observation
        for e in self.edges:
            if e["src"] == src and e["dst"] == dst and e["relation"] == relation:
                if weight > e["weight"]:
                    e["weight"], e["fact"] = round(weight, 4), fact
                return
        self.edges.append({"src": src, "dst": dst, "relation": relation,
                           "weight": round(weight, 4), "fact": fact})

    def record_run(self, dataset_name: str, target_column: str,
                   shap_importance: Optional[List[Tuple[str, float]]] = None,
                   correlations: Optional[List[Tuple[str, str, float]]] = None,
                   task_type: str = ""):
        """
        Ingest one pipeline run's structured facts. Called by ExplainerAgent
        after SHAP analysis completes.

        Args:
            shap_importance: [(feature, mean_abs_shap), ...] strongest first
            correlations:    [(col_a, col_b, correlation_or_vif), ...]
        """
        ds = self._add_node("dataset", dataset_name)
        tgt = self._add_node("target", target_column)
        self._add_edge(tgt, ds, "belongs_to", 1.0,
                       f"'{target_column}' is the prediction target of the "
                       f"'{dataset_name}' dataset ({task_type}).")

        for feature, shap_val in (shap_importance or [])[:15]:
            col = self._add_node("column", feature)
            self._add_edge(col, ds, "belongs_to", 1.0,
                           f"'{feature}' is a column in '{dataset_name}'.")
            self._add_edge(col, tgt, "predicts", abs(float(shap_val)),
                           f"In '{dataset_name}', '{feature}' was a driver of "
                           f"'{target_column}' (mean |SHAP| = {abs(float(shap_val)):.4f}).")

        for col_a, col_b, strength in (correlations or []):
            a = self._add_node("column", col_a)
            b = self._add_node("column", col_b)
            fact = (f"In '{dataset_name}', '{col_a}' and '{col_b}' are strongly "
                    f"related (strength = {abs(float(strength)):.2f}).")
            self._add_edge(a, b, "correlates_with", abs(float(strength)), fact)
            self._add_edge(b, a, "correlates_with", abs(float(strength)), fact)

        self.save()

    # ── Querying ──────────────────────────────────────────────────────────

    def _match_entities(self, question: str) -> Set[str]:
        """
        Fuzzy entity linking: a node matches if its full name appears in the
        question, or (for snake_case columns) if all its word-parts do.
        """
        q = question.lower()
        q_tokens = set(re.findall(r"[a-z0-9_]+", q))
        matched = set()
        for nid, node in self.nodes.items():
            name = node["name"].lower()
            if name in q:
                matched.add(nid)
                continue
            parts = [p for p in re.split(r"[_\s]+", name) if len(p) > 2]
            if parts and all(p in q_tokens for p in parts):
                matched.add(nid)
        return matched

    def query(self, question: str, max_hops: int = 2,
              max_facts: int = 8) -> List[str]:
        """
        Multi-hop retrieval: BFS from every entity mentioned in the question,
        collecting edge facts up to max_hops away. Facts are ranked by
        (fewer hops first, then edge weight) so direct evidence beats distant.

        Returns a list of plain-English fact sentences (possibly empty).
        """
        seeds = self._match_entities(question)
        if not seeds:
            return []

        # adjacency (undirected walk — relations carry direction in the fact text)
        adj: Dict[str, List[Dict]] = {}
        for e in self.edges:
            adj.setdefault(e["src"], []).append(e)
            adj.setdefault(e["dst"], []).append(e)

        collected: List[Tuple[int, float, str]] = []  # (hops, -weight, fact)
        visited: Set[str] = set(seeds)
        frontier = deque((nid, 0) for nid in seeds)

        while frontier:
            nid, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            for edge in adj.get(nid, []):
                other = edge["dst"] if edge["src"] == nid else edge["src"]
                collected.append((depth + 1, -edge["weight"], edge["fact"]))
                if other not in visited:
                    visited.add(other)
                    frontier.append((other, depth + 1))

        # Rank: nearest hops first, then strongest edges; dedupe fact strings
        collected.sort()
        seen, facts = set(), []
        for _, _, fact in collected:
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
            if len(facts) >= max_facts:
                break
        return facts

    def stats(self) -> Dict:
        types = {}
        for n in self.nodes.values():
            types[n["type"]] = types.get(n["type"], 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "node_types": types}


if __name__ == "__main__":
    import sys
    kg = KnowledgeGraph()
    print(f"Graph: {kg.stats()}")
    q = " ".join(sys.argv[1:]) or "how does monthly_charges relate to churn"
    for fact in kg.query(q):
        print(f"  - {fact}")

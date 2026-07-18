"""
DSA Engine -- RGCN data sources
===============================
One place that knows how to read problem vectors + tags and curated concept
edges from either a live database or a file. The rest of the RGCN code is
source-agnostic.

Problems  (vectors + tags + similar ids):
    load_problems_qdrant()   <- DB (default)
    load_problems_parquet()  <- file fallback

Curated concept edges  (problem->topic, topic<->topic, topic text):
    load_curated_neo4j()       <- graph DB
    load_curated_normalized()  <- JSON files
    (GRAPH_SOURCE="tags" needs neither -- tags from the problem payload suffice)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import config as C

TAG_KEYS = ("topic_tags", "algorithm_tags", "data_structure_tags",
            "patterns", "techniques")
# non-vector payload fields worth carrying into the output Qdrant collections
META_KEYS = ("problem_id", "title", "title_slug", "difficulty_score",
             "topic_tags", "algorithm_tags", "data_structure_tags",
             "patterns", "techniques", "skill_tags", "companies")


@dataclass
class ProblemData:
    problem_id: str
    title_slug: str
    feature: np.ndarray
    tags: dict[str, list]
    similar_ids: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    try:
        return [v for v in list(val) if v is not None and str(v).strip()]
    except Exception:
        return []


def _as_vec(val):
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0 or np.any(np.isnan(arr)):
            return None
        return arr
    except Exception:
        return None


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:   # tolerate BOM
        return json.load(f)


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

def load_problems() -> list[ProblemData]:
    if C.FEATURE_SOURCE == "qdrant":
        return load_problems_qdrant()
    return load_problems_parquet()


def load_problems_qdrant() -> list[ProblemData]:
    """Scroll the embedder's Qdrant collection: vector -> feature, payload -> tags."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=C.QDRANT_URL, api_key=C.QDRANT_API_KEY, timeout=30)
    coll = C.QDRANT_SOURCE_COLLECTION
    try:
        info = client.get_collection(coll)
    except Exception as e:
        raise RuntimeError(
            f"Cannot read Qdrant collection '{coll}' at {C.QDRANT_URL}: "
            f"{e.__class__.__name__}: {str(e)[:120]}\n"
            f"Run embedder.py with --qdrant-url/--collection first, or use "
            f"RGCN_FEATURE_SOURCE=parquet."
        )
    total = client.count(collection_name=coll).count
    print(f"[->] reading {total} points from Qdrant '{coll}' "
          f"(dim {info.config.params.vectors.size})")

    out, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=coll, limit=512,
            with_vectors=True, with_payload=True, offset=offset,
        )
        for p in batch:
            vec = p.vector
            if isinstance(vec, dict):                  # named vectors -> take first
                vec = next(iter(vec.values()), None)
            feature = _as_vec(vec)
            if feature is None:
                continue
            pl = p.payload or {}
            out.append(ProblemData(
                problem_id=str(pl.get("problem_id", p.id)),
                title_slug=str(pl.get("title_slug", "") or ""),
                feature=feature,
                tags={k: _as_list(pl.get(k)) for k in TAG_KEYS},
                similar_ids=_as_list(pl.get("similar_problem_ids")),
                meta={k: pl.get(k) for k in META_KEYS if k in pl},
            ))
        if offset is None:
            break
    print(f"[OK] {len(out)} problems loaded from Qdrant")
    return out


def load_problems_parquet() -> list[ProblemData]:
    """File fallback: read vector_pool_embedded.parquet."""
    import pandas as pd
    if not C.INPUT_PARQUET.exists():
        raise FileNotFoundError(f"{C.INPUT_PARQUET} not found "
                                f"(and FEATURE_SOURCE=parquet).")
    df = pd.read_parquet(C.INPUT_PARQUET)
    col = C.PROBLEM_FEATURE_COL
    if col not in df.columns:
        raise KeyError(f"'{col}' not in parquet columns {list(df.columns)}")
    print(f"[->] reading {len(df)} rows from {C.INPUT_PARQUET.name}")
    out = []
    for _, row in df.iterrows():
        feature = _as_vec(row.get(col))
        if feature is None:
            continue
        out.append(ProblemData(
            problem_id=str(row.get("problem_id", "")),
            title_slug=str(row.get("title_slug", "") or ""),
            feature=feature,
            tags={k: _as_list(row.get(k)) for k in TAG_KEYS},
            similar_ids=_as_list(row.get("similar_problem_ids")),
            meta={k: (row[k].tolist() if isinstance(row.get(k), np.ndarray)
                      else row.get(k))
                  for k in META_KEYS if k in df.columns},
        ))
    print(f"[OK] {len(out)} problems loaded from parquet")
    return out


# ---------------------------------------------------------------------------
# Curated concept edges
# ---------------------------------------------------------------------------

@dataclass
class CuratedEdges:
    problem_topic: list[tuple[str, str]]                 # (problem_slug, topic_slug)
    topic_topic:   list[tuple[str, str, float, int]]     # (a, b, jaccard, shared)
    topic_text:    dict[str, str]                        # topic_slug -> name[: desc]


def load_curated() -> CuratedEdges | None:
    if C.GRAPH_SOURCE == "neo4j":
        return load_curated_neo4j()
    if C.GRAPH_SOURCE == "normalized":
        return load_curated_normalized()
    return None


def load_curated_normalized() -> CuratedEdges:
    if not C.GRAPH_PROBLEM_TOPIC.exists():
        raise FileNotFoundError(
            f"GRAPH_SOURCE=normalized but missing {C.GRAPH_PROBLEM_TOPIC}\n"
            f"Drop the JSONs in {C.NORMALIZED_GRAPH_DIR} or set RGCN_GRAPH_DIR."
        )

    def _src_tgt_pt(e):
        """
        Handle all known problem-topic JSON field name variants:
          normalized zip : source / target
          raw   : source / target  (same -- confirmed from actual file)
          potential alts : title_slug / topic_slug, problem_slug / topic_slug,
                           problem_id / topic_id  (Greptile P1: previously missing)
        Checking every known variant ensures curated HAS_TOPIC edges are
        never silently dropped when the file format changes between runs.
        """
        src = (e.get("source") or e.get("title_slug")
               or e.get("problem_slug") or e.get("problem_id") or "")
        tgt = (e.get("target") or e.get("topic_slug")
               or e.get("topic_id") or "")
        return str(src), str(tgt)

    raw_pt = _load_json(C.GRAPH_PROBLEM_TOPIC)
    pt = []
    for e in raw_pt:
        src, tgt = _src_tgt_pt(e)
        if src and tgt:
            pt.append((src, tgt))
    if not pt and raw_pt:
        sample_keys = list(raw_pt[0].keys()) if raw_pt else []
        print(f"  [WARN] problem_topic file has {len(raw_pt)} records but 0 edges. "
              f"Sample keys: {sample_keys}")

    tt = []
    if C.GRAPH_TOPIC_TOPIC.exists():
        for e in _load_json(C.GRAPH_TOPIC_TOPIC):
            src = str(e.get("source") or e.get("source_topic_id")
                      or e.get("topic_slug_a") or "")
            tgt = str(e.get("target") or e.get("target_topic_id")
                      or e.get("topic_slug_b") or "")
            if src and tgt:
                tt.append((src, tgt,
                           float(e.get("jaccard", 0.0)),
                           int(e.get("shared_problem_count", 0))))
    text = {}
    if C.GRAPH_TOPIC_NODES.exists():
        for n in _load_json(C.GRAPH_TOPIC_NODES):
            slug = str(n.get("topic_slug") or n.get("topic_id") or "")
            text[slug] = _topic_text(n)
    print(f"[OK] curated (normalized JSON): {len(pt)} HAS_TOPIC, {len(tt)} CO_OCCURS")
    return CuratedEdges(pt, tt, text)


def load_curated_neo4j() -> CuratedEdges:
    """Read the same curated edges live from Neo4j."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("GRAPH_SOURCE=neo4j needs the driver: "
                          "uv pip install neo4j")
    driver = GraphDatabase.driver(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    pt, tt, text = [], [], {}
    with driver.session(database=C.NEO4J_DATABASE) as s:
        for r in s.run(
            "MATCH (p:Problem)-[:HAS_TOPIC]->(t:Topic) "
            "RETURN p.title_slug AS p, t.topic_slug AS t"
        ):
            if r["p"] and r["t"]:
                pt.append((str(r["p"]), str(r["t"])))
        for r in s.run(
            "MATCH (a:Topic)-[r:CO_OCCURS_WITH]->(b:Topic) "
            "RETURN a.topic_slug AS a, b.topic_slug AS b, "
            "r.jaccard AS j, r.shared_problem_count AS s"
        ):
            if r["a"] and r["b"]:
                tt.append((str(r["a"]), str(r["b"]),
                           float(r["j"] or 0.0), int(r["s"] or 0)))
        for r in s.run(
            "MATCH (t:Topic) RETURN t.topic_slug AS slug, "
            "t.topic_name AS name, t.short_description AS desc"
        ):
            if r["slug"]:
                text[str(r["slug"])] = _topic_text(
                    {"topic_slug": r["slug"], "topic_name": r["name"],
                     "short_description": r["desc"]})
    driver.close()
    print(f"[OK] curated (Neo4j): {len(pt)} HAS_TOPIC, {len(tt)} CO_OCCURS")
    if not pt:
        print("[!] Neo4j returned 0 HAS_TOPIC edges -- graph DB may be schema-only. "
              "Falling back to tag-derived concepts only.")
    return CuratedEdges(pt, tt, text)


def _topic_text(n: dict) -> str:
    slug = str(n.get("topic_slug") or n.get("topic_id") or "")
    name = str(n.get("topic_name", "") or slug).strip()
    desc = str(n.get("short_description", "") or "").strip()
    # most descriptions are "DSA topic: x" placeholders -> name carries the signal
    return f"{name}: {desc}" if desc and not desc.lower().startswith("dsa topic") else name
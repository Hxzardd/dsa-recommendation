
"""
DSA Engine -- RGCN Step 1: Build the heterogeneous graph
========================================================
Assembles a two-node-type, four-relation heterogeneous graph. Problem vectors
and tags come from `sources` (Qdrant by default, parquet fallback); curated
concept edges come from the graph DB / JSON when GRAPH_SOURCE != "tags".

Nodes
    problem : feature = embedder vector (config.PROBLEM_FEATURE_COL)
    concept : feature = centroid of member problems (default) or text embedding

Relations (each has its own weight matrix in the model)
    (problem, uses,     concept)   tag edges + curated HAS_TOPIC edges, weighted
    (concept, used_by,  problem)   reverse of `uses`
    (problem, similar,  problem)   similar_problem_ids + cosine-KNN
    (concept, cooccurs, concept)   parquet/payload co-occurrence + curated jaccard

Output: config.GRAPH_PATH (.pt) + config.CONCEPT_INDEX (json).

Run:
    python rgcn/build_graph.py
    RGCN_GRAPH_SOURCE=normalized python rgcn/build_graph.py
    RGCN_GRAPH_SOURCE=neo4j      python rgcn/build_graph.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import config as C
import sources as S


def _l2(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return (mat / n).astype(np.float32)


def _norm_concept(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Concept vocabulary + problem->concept edges
# ---------------------------------------------------------------------------

def _build_concepts_and_uses(problems, slug_to_idx, pid_to_idx, curated):
    uses_w: dict[tuple[int, int], float] = {}
    concept_members: dict[int, set] = defaultdict(set)
    concept_text: dict[str, str] = {}
    concept_to_idx: dict[str, int] = {}
    stats = {"tag_edges": 0, "curated_edges": 0, "curated_unresolved": 0}

    def _cid(name: str) -> int:
        key = _norm_concept(name)
        if key not in concept_to_idx:
            concept_to_idx[key] = len(concept_to_idx)
        return concept_to_idx[key]

    def _add(p_idx: int, c_idx: int, w: float):
        k = (p_idx, c_idx)
        uses_w[k] = w if k not in uses_w else max(uses_w[k], w)
        concept_members[c_idx].add(p_idx)

    # (1) tag edges from the problem payload/columns -- always present
    for p_idx, pr in enumerate(problems):
        for col, w in C.CONCEPT_TAG_WEIGHTS.items():
            for tag in pr.tags.get(col, []):
                c = _cid(tag)
                _add(p_idx, c, w)
                concept_text.setdefault(_norm_concept(tag), str(tag).replace("_", " "))
                stats["tag_edges"] += 1

    # (2) curated HAS_TOPIC edges from graph DB / JSON
    if curated is not None:
        for slug, txt in curated.topic_text.items():
            concept_text.setdefault(_norm_concept(slug), txt)
        for src_slug, tgt_slug in curated.problem_topic:
            if not tgt_slug:
                continue
            p_idx = slug_to_idx.get(src_slug) or pid_to_idx.get(src_slug)
            if p_idx is None:
                stats["curated_unresolved"] += 1
                continue
            c = _cid(tgt_slug)
            _add(p_idx, c, C.NORMALIZED_EDGE_WEIGHT)
            concept_text.setdefault(_norm_concept(tgt_slug),
                                    _norm_concept(tgt_slug).replace("_", " "))
            stats["curated_edges"] += 1

    # remap to alphabetical concept order for stable artifacts
    ordered = sorted(concept_to_idx)
    remap = {concept_to_idx[c]: i for i, c in enumerate(ordered)}
    uses_w = {(p, remap[c]): w for (p, c), w in uses_w.items()}
    concept_members = {remap[c]: sorted(v) for c, v in concept_members.items()}
    concept_to_idx = {c: i for i, c in enumerate(ordered)}
    return ordered, concept_to_idx, uses_w, concept_members, concept_text, stats


# ---------------------------------------------------------------------------
# Concept<->concept edges
# ---------------------------------------------------------------------------

def _build_cooccur(problems, concept_to_idx, curated):
    pair_w: dict[tuple[int, int], float] = {}

    counts: Counter = Counter()
    for pr in problems:
        cs = sorted({concept_to_idx[_norm_concept(t)]
                     for col in C.CONCEPT_TAG_WEIGHTS for t in pr.tags.get(col, [])
                     if _norm_concept(t) in concept_to_idx})
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                counts[(cs[i], cs[j])] += 1
    if counts:
        mx = np.log1p(max(counts.values()))
        for (a, b), c in counts.items():
            if c >= C.COOCCUR_MIN_COUNT:
                pair_w[(a, b)] = max(pair_w.get((a, b), 0.0), float(np.log1p(c) / mx))

    if curated is not None and curated.topic_topic:
        jmax = max((j for _a, _b, j, _s in curated.topic_topic), default=1.0) or 1.0
        for a_s, b_s, j, shared in curated.topic_topic:
            if shared < C.TT_MIN_SHARED_PROBLEMS:
                continue
            a = concept_to_idx.get(_norm_concept(a_s))
            b = concept_to_idx.get(_norm_concept(b_s))
            if a is None or b is None or a == b:
                continue
            k = (min(a, b), max(a, b))
            pair_w[k] = max(pair_w.get(k, 0.0), j / jmax)

    if not pair_w:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    src, dst, w = [], [], []
    for (a, b), weight in pair_w.items():
        src += [a, b]; dst += [b, a]; w += [weight, weight]
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(w, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_graph() -> dict:
    print("\n" + "=" * 62)
    print("  RGCN STEP 1 -- BUILD HETEROGENEOUS GRAPH")
    print("=" * 62)
    print(C.summary())

    problems = S.load_problems()
    if not problems:
        raise ValueError("No problems loaded from the configured source.")
    dims = {pr.feature.shape[0] for pr in problems}
    if len(dims) > 1:
        raise ValueError(f"Inconsistent problem feature dims: {dims}")
    feat_dim = dims.pop()

    X_problem = _l2(np.vstack([pr.feature for pr in problems]))
    n_problems = X_problem.shape[0]
    prob_ids   = [pr.problem_id for pr in problems]
    prob_slugs = [pr.title_slug for pr in problems]
    pid_to_idx  = {p: i for i, p in enumerate(prob_ids)}
    slug_to_idx = {s: i for i, s in enumerate(prob_slugs) if s}
    print(f"[OK] {n_problems} problem nodes  (feature dim {feat_dim})")

    curated = S.load_curated()  # None for GRAPH_SOURCE=tags

    (concepts, concept_to_idx, uses_w, concept_members,
     concept_text, stats) = _build_concepts_and_uses(
        problems, slug_to_idx, pid_to_idx, curated)
    n_concepts = len(concepts)
    print(f"[OK] {n_concepts} concept nodes  (graph source='{C.GRAPH_SOURCE}')")
    if curated is not None:
        print(f"     curated problem->concept edges : {stats['curated_edges']} "
              f"({stats['curated_unresolved']} unresolved slugs skipped)")
    print(f"     tag-derived edges              : {stats['tag_edges']}")

    uses_src = torch.tensor([k[0] for k in uses_w], dtype=torch.long)
    uses_dst = torch.tensor([k[1] for k in uses_w], dtype=torch.long)
    uses_weight = torch.tensor(list(uses_w.values()), dtype=torch.float32)
    print(f"[OK] {uses_src.numel()} problem->concept edges (deduped) "
          f"avg {uses_src.numel()/max(n_problems,1):.2f}/problem")

    if C.CONCEPT_FEATURE_MODE == "text":
        X_concept = _concept_text_features(concepts, concept_text)
    else:
        X_concept = _concept_centroid_features(concepts, concept_members, X_problem)
    print(f"[OK] concept features: mode='{C.CONCEPT_FEATURE_MODE}' shape={tuple(X_concept.shape)}")

    cc_index, cc_weight = _build_cooccur(problems, concept_to_idx, curated)
    print(f"[OK] {cc_index.shape[1]} concept<->concept edges")

    # similar edges
    sim_pairs: set[tuple[int, int]] = set()
    declared = 0
    for p_idx, pr in enumerate(problems):
        for sid in pr.similar_ids:
            j = slug_to_idx.get(str(sid)) or pid_to_idx.get(str(sid))
            if j is not None and j != p_idx:
                sim_pairs.add((min(p_idx, j), max(p_idx, j)))
                declared += 1
    knn = _add_knn_edges(X_problem, C.SIMILARITY_KNN_K, sim_pairs) \
        if C.SIMILARITY_KNN_K > 0 and n_problems > 1 else 0
    sim_src, sim_dst = [], []
    for a, b in sim_pairs:
        sim_src += [a, b]; sim_dst += [b, a]
    sim_index = torch.tensor([sim_src, sim_dst], dtype=torch.long) if sim_src \
        else torch.empty((2, 0), dtype=torch.long)
    print(f"[OK] {sim_index.shape[1]} problem<->problem similar edges "
          f"({declared} declared, +{knn} KNN k={C.SIMILARITY_KNN_K})")

    graph = {
        "node_features": {"problem": torch.from_numpy(X_problem),
                          "concept": torch.from_numpy(X_concept)},
        "num_nodes": {"problem": n_problems, "concept": n_concepts},
        "relations": {
            "uses":     ("problem", "concept", torch.stack([uses_src, uses_dst]), uses_weight),
            "used_by":  ("concept", "problem", torch.stack([uses_dst, uses_src]), uses_weight),
            "similar":  ("problem", "problem", sim_index, None),
            "cooccurs": ("concept", "concept", cc_index, cc_weight),
        },
        "problem_ids": prob_ids,
        "problem_slugs": prob_slugs,
        "problem_meta": [pr.meta for pr in problems],   # for DB-fed output payloads
        "problem_features": torch.from_numpy(X_problem),  # for full_embedding concat
        "concepts": concepts,
        "meta": {
            "problem_feat_dim": int(feat_dim),
            "concept_feat_dim": int(X_concept.shape[1]),
            "feature_source": C.FEATURE_SOURCE,
            "feature_col": C.PROBLEM_FEATURE_COL,
            "concept_mode": C.CONCEPT_FEATURE_MODE,
            "graph_source": C.GRAPH_SOURCE,
            "stats": stats,
        },
    }

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save tensors as .npz (no pickle, safe to load) ───────────────────────
    graph_npz_path = C.ARTIFACTS_DIR / "graph_tensors.npz"
    np.savez(
        graph_npz_path,
        problem_features = X_problem,
        concept_features = X_concept,
        uses_src         = uses_src.numpy(),
        uses_dst         = uses_dst.numpy(),
        uses_weight      = uses_weight.numpy(),
        sim_index        = sim_index.numpy(),
        cc_index         = cc_index.numpy(),
        cc_weight        = cc_weight.numpy() if cc_weight is not None
                           else np.array([]),
    )

    # ── Save non-tensor metadata as JSON (human-readable, no pickle) ─────────
    import json as _json
    graph_meta_path = C.ARTIFACTS_DIR / "graph_meta.json"
    # problem_meta contains dicts of python primitives only
    _safe_meta = []
    for m in [pr.meta for pr in problems]:
        safe = {}
        for k, v in (m or {}).items():
            if isinstance(v, (str, int, float, bool, type(None))):
                safe[k] = v
            elif isinstance(v, (list, tuple)):
                safe[k] = [x for x in v if isinstance(x, (str, int, float, bool))]
        _safe_meta.append(safe)

    # Content fingerprint: hash of the sorted problem_id list. train_rgcn.py
    # writes this same hash into model_arch.json; ingest_rgcn_to_qdrant.py
    # checks both match before writing to Qdrant, catching the case where
    # row counts coincidentally match but the underlying problem set changed.
    import hashlib as _hashlib
    _graph_fingerprint = _hashlib.sha256(
        "|".join(sorted(prob_ids)).encode()
    ).hexdigest()[:16]

    _json.dump({
        "problem_ids":   prob_ids,
        "problem_slugs": prob_slugs,
        "problem_meta":  _safe_meta,
        "concepts":      concepts,
        "num_nodes":     {"problem": n_problems, "concept": n_concepts},
        "graph_fingerprint": _graph_fingerprint,
        "meta": {
            "problem_feat_dim":  int(feat_dim),
            "concept_feat_dim":  int(X_concept.shape[1]),
            "feature_source":    C.FEATURE_SOURCE,
            "feature_col":       C.PROBLEM_FEATURE_COL,
            "concept_mode":      C.CONCEPT_FEATURE_MODE,
            "graph_source":      C.GRAPH_SOURCE,
            "stats":             stats,
        },
    }, open(graph_meta_path, "w", encoding="utf-8"), indent=2)

    # ── Keep graph dict in memory for _verify and return ─────────────────────
    # Also write the legacy .pt ONLY as a local scratch file (not committed).
    # Loaders should use graph_tensors.npz + graph_meta.json instead.
    torch.save(graph, C.GRAPH_PATH)   # scratch only — do NOT commit this file
    with open(C.CONCEPT_INDEX, "w", encoding="utf-8") as f:
        json.dump({c: i for i, c in enumerate(concepts)}, f, indent=2)
    print(f"\n[OK] graph saved   -> {C.GRAPH_PATH}")
    print(f"[OK] concepts saved-> {C.CONCEPT_INDEX}")
    _verify(graph)
    return graph


# ---------------------------------------------------------------------------
# Concept features
# ---------------------------------------------------------------------------

def _concept_centroid_features(concepts, members, X_problem) -> np.ndarray:
    out = np.zeros((len(concepts), X_problem.shape[1]), dtype=np.float32)
    for c_idx in range(len(concepts)):
        idxs = members.get(c_idx, [])
        if idxs:
            out[c_idx] = X_problem[idxs].mean(axis=0)
    return _l2(out)


def _concept_text_features(concepts, concept_text) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"[->] Loading {C.CONCEPT_TEXT_MODEL} for concept text features...")
    model = SentenceTransformer(C.CONCEPT_TEXT_MODEL)
    texts = [concept_text.get(c, c.replace("_", " ")) for c in concepts]
    return model.encode(texts, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)


def _add_knn_edges(X: np.ndarray, k: int, sim_pairs: set) -> int:
    added = 0
    Xt = torch.from_numpy(X)
    n = Xt.shape[0]
    for start in range(0, n, 512):
        block = Xt[start:start + 512]
        topk = torch.topk(block @ Xt.T, k=min(k + 1, n), dim=1).indices
        for bi in range(block.shape[0]):
            i = start + bi
            for j in topk[bi].tolist():
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in sim_pairs:
                    sim_pairs.add((a, b)); added += 1
    return added


# ---------------------------------------------------------------------------
# Verification gate
# ---------------------------------------------------------------------------

def _verify(graph: dict) -> None:
    print("\n-- STEP 1 VERIFICATION ------------------------------------")
    np_ = graph["num_nodes"]["problem"]; nc_ = graph["num_nodes"]["concept"]
    print(f"  feature source: {graph['meta']['feature_source']}")
    print(f"  graph source  : {graph['meta']['graph_source']}")
    print(f"  problem nodes : {np_}")
    print(f"  concept nodes : {nc_}")
    ok = True
    for rel, (st, dt, ei, ew) in graph["relations"].items():
        e = ei.shape[1]
        if e:
            in_bounds = ei[0].max().item() < graph["num_nodes"][st] and \
                        ei[1].max().item() < graph["num_nodes"][dt]
            ok &= in_bounds
            flag = "OK" if in_bounds else "OUT-OF-BOUNDS"
        else:
            flag = "empty"
        print(f"  {rel:<9} {st:>7}->{dt:<7} edges={e:<7} "
              f"weight={(ew.numel() if ew is not None else 'none')}  [{flag}]")
    deg = torch.zeros(np_, dtype=torch.long)
    used = graph["relations"]["uses"][2]
    deg.index_add_(0, used[0], torch.ones(used.shape[1], dtype=torch.long))
    print(f"  problems with zero concept edges: {int((deg==0).sum())}")
    feat_ok = not torch.any(torch.isnan(graph["node_features"]["problem"])) \
        and not torch.any(torch.isnan(graph["node_features"]["concept"]))
    print(f"  features finite: {feat_ok}")
    print(f"  RESULT: {'PASS' if (ok and feat_ok) else 'FAIL'}")
    print("-" * 58)


if __name__ == "__main__":
    build_graph()

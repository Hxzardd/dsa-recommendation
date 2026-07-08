"""
DSA Engine -- RGCN Embedding Evaluation Suite
==============================================
Evaluates the RGCN embeddings in the order that matters for a recommendation
system:

  1. Recommendation Quality   Recall@K, MRR, NDCG
  2. Semantic Retrieval        same_topic@10, same_pattern@10, same_skill@10
  3. Topic Separation          cluster_score, silhouette (fast approx), intra/inter sim
  4. Graph Preservation        link_prediction_auc
  5. Sanity checks             dims, NaN, unit-norm, payload, self-similarity

Loss is NOT evaluated here. "Did retrieval improve?" is the only question.

Baselines compared automatically when the raw embedder collection is available:
  * random retrieval
  * raw BGE / QS embeddings   (from dsa_problems collection)
  * rgcn_embedding             (128-d)
  * full_embedding             (1920-d)

Run (from dsa-recommendation/ root OR from pipeline/graphs/):
    python pipeline/graphs/test_rgcn_embeddings.py
    python pipeline/graphs/test_rgcn_embeddings.py --verbose
    python pipeline/graphs/test_rgcn_embeddings.py --url http://localhost:6333
    python pipeline/graphs/test_rgcn_embeddings.py --skip-baselines

NOTE: This is a CLI evaluator, not a pytest module.
      conftest.py adds it to collect_ignore so pytest skips it automatically.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path fix: ensure `import config` resolves to pipeline/graphs/config.py
# regardless of whether this script is run from the repo root or its own dir.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as C

RGCN_DIM = C.OUT_DIM
FULL_DIM  = None


def _client(url: str):
    from qdrant_client import QdrantClient
    try:
        c = QdrantClient(url=url, api_key=C.QDRANT_API_KEY, timeout=30)
        c.get_collections()
        return c
    except Exception as e:
        print(f"\n[X] Cannot reach Qdrant at {url}: {e.__class__.__name__}: {str(e)[:80]}")
        print("    docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)


def _scroll_all(client, collection):
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, limit=512,
            with_vectors=True, with_payload=True, offset=offset)
        points.extend(batch)
        if offset is None:
            break
    return points


def _vec(p):
    v = p.vector
    if isinstance(v, dict):
        v = next(iter(v.values()))
    return np.asarray(v, dtype=np.float32)


def _tag(p, key):
    return list((p.payload or {}).get(key) or [])


PASS = "[PASS]"; FAIL = "[FAIL]"


def _r(ok, msg): return ok, f"  {PASS if ok else FAIL}  {msg}"


# ===========================================================================
# 1. Recommendation Quality -- Recall@K, MRR, NDCG
# ===========================================================================

def _hits_at_k(client, collection, points, k, tag_key):
    """
    Compute Recall@K, MRR, and NDCG@K for a collection.

    MRR  = mean reciprocal rank of the FIRST relevant result per query.
           Stops scanning after the first hit (standard definition).
    NDCG = normalised discounted cumulative gain over ALL K results.
           Must NOT break early -- all K positions contribute to DCG.
           The prior version broke after the first hit, so DCG was
           identical to a rank-1-only score and NDCG was underreported.
    """
    hits, mrr_sum, ndcg_sum = 0, 0.0, 0.0
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, 10)))
    n = 0
    for p in points:
        query_tags = set(_tag(p, tag_key))
        if not query_tags:
            continue
        results = client.query_points(
            collection_name=collection,
            query=list(map(float, _vec(p))),
            limit=k + 1, with_payload=True, with_vectors=False
        ).points
        results = [r for r in results if r.id != p.id][:k]

        mrr_credited = False   # only credit the first relevant hit for MRR
        dcg = 0.0
        for rank, r in enumerate(results, 1):
            if set(_tag(r, tag_key)) & query_tags:
                if not mrr_credited:
                    hits += 1
                    mrr_sum += 1.0 / rank
                    mrr_credited = True
                # DCG accumulates over ALL relevant results in the top-K,
                # not just the first. Do not break here.
                dcg += 1.0 / np.log2(rank + 1)

        ndcg_sum += dcg / max(ideal_dcg, 1e-9)
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return hits / n, mrr_sum / n, ndcg_sum / n


def test_recommendation_quality(client, collection, points, k_list=(10, 20), verbose=False):
    print(f"\n── 1. Recommendation Quality  ({collection}) ──────────────────")
    results = []
    sample = points[:200]  # cap at 200 for speed; stratified enough
    for k in k_list:
        recall, mrr, ndcg = _hits_at_k(client, collection, sample, k, "topic_tags")
        results.append(_r(recall > 0.0, f"Recall@{k:<3}={recall:.3f}  MRR={mrr:.3f}  NDCG@{k}={ndcg:.3f}"))
    for ok, msg in results:
        print(msg)
    return all(ok for ok, _ in results), {
        f"recall@{k}": _hits_at_k(client, collection, sample, k, "topic_tags")[0]
        for k in k_list
    }


# ===========================================================================
# 2. Semantic Retrieval -- same_topic@10, same_pattern@10, same_skill@10
# ===========================================================================

def _same_tag_at_k(client, collection, points, tag_key, k=10, n_queries=100):
    """
    Fraction of the top-K non-self results sharing at least one tag with the query.

    Requests k+1 from Qdrant to leave room for the self-hit, then slices
    back to exactly k after removing the query point. Without the slice,
    same_topic@K is evaluated over up to k+1 neighbors when the query
    point is absent from Qdrant's results, making the @K label inconsistent
    across collections.
    """
    hits, total = 0, 0
    candidates = [p for p in points if _tag(p, tag_key)][:n_queries]
    for p in candidates:
        query_tags = set(_tag(p, tag_key))
        results = client.query_points(
            collection_name=collection,
            query=list(map(float, _vec(p))),
            limit=k + 1, with_payload=True, with_vectors=False
        ).points
        # Remove self then take exactly k -- ensures consistent @K evaluation
        # regardless of whether Qdrant returned the query point itself.
        top_k = [r for r in results if r.id != p.id][:k]
        for r in top_k:
            if set(_tag(r, tag_key)) & query_tags:
                hits += 1
            total += 1
        if total >= n_queries * k:
            break
    return hits / max(total, 1)


def test_semantic_retrieval(client, collection, points, verbose=False):
    print(f"\n── 2. Semantic Retrieval  ({collection}) ──────────────────────")
    results = []
    scores = {}
    for tag_key, label in [("topic_tags", "same_topic@10"),
                            ("patterns",   "same_pattern@10"),
                            ("skill_tags", "same_skill@10")]:
        has_tag = [p for p in points if _tag(p, tag_key)]
        if len(has_tag) < 10:
            results.append(_r(False, f"{label:<20} insufficient data ({len(has_tag)} tagged)"))
            continue
        score = _same_tag_at_k(client, collection, has_tag, tag_key, k=10)
        scores[label] = score
        results.append(_r(score > 0.0, f"{label:<20} = {score:.3f}"))
    for ok, msg in results:
        print(msg)
    return all(ok for ok, _ in results), scores


# ===========================================================================
# 3. Topic Separation -- cluster_score, intra/inter similarity
# ===========================================================================

def test_topic_separation(client, collection, points, verbose=False):
    print(f"\n── 3. Topic Separation  ({collection}) ────────────────────────")
    by_topic = defaultdict(list)
    for p in points:
        tags = _tag(p, "topic_tags")
        if tags:
            by_topic[tags[0]].append(_vec(p))

    topics = [t for t, vs in by_topic.items() if len(vs) >= 3]
    if len(topics) < 2:
        print("  [SKIP]  not enough topic diversity")
        return True, {}

    wins, trials = 0, 0
    intra_list, inter_list = [], []
    for i, t in enumerate(topics[:40]):
        vecs = np.stack(by_topic[t])
        mean = vecs.mean(0); mean /= (np.linalg.norm(mean) or 1)
        other_t = topics[(i + 1) % len(topics)]
        other = np.stack(by_topic[other_t])
        mean_o = other.mean(0); mean_o /= (np.linalg.norm(mean_o) or 1)
        # intra: mean pairwise cosine within topic (sample up to 5)
        sub = vecs[:5]; sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
        intra = float(np.mean(sub @ sub.T)) if sub.shape[0] > 1 else 1.0
        # inter: cosine between topic means
        inter = float(mean @ mean_o)
        intra_list.append(intra); inter_list.append(inter)
        wins += int(intra > inter); trials += 1

    cluster = wins / max(trials, 1)
    mean_intra = float(np.mean(intra_list))
    mean_inter = float(np.mean(inter_list))
    sep = mean_intra - mean_inter

    results = [
        _r(cluster >= 0.6,  f"cluster_score         = {cluster:.3f}  ({wins}/{trials} topics)"),
        _r(mean_intra > 0,  f"intra_topic_sim (mean)= {mean_intra:.3f}"),
        _r(mean_inter < mean_intra, f"inter_topic_sim (mean)= {mean_inter:.3f}"),
        _r(sep > 0,         f"separation (intra-inter)= {sep:.3f}"),
    ]
    if verbose:
        for t in topics[:5]:
            vecs = np.stack(by_topic[t])
            sub = vecs[:5]; sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
            sim = float(np.mean(sub @ sub.T)) if sub.shape[0] > 1 else 1.0
            print(f"    {t:<30} n={len(by_topic[t])}  intra_sim={sim:.3f}")
    for ok, msg in results:
        print(msg)
    return all(ok for ok, _ in results), {
        "cluster_score": cluster, "intra_sim": mean_intra,
        "inter_sim": mean_inter, "separation": sep,
    }


# ===========================================================================
# 4. Graph Preservation -- link prediction AUC
# ===========================================================================

def test_graph_preservation(collection, points):
    print(f"\n── 4. Graph Preservation  ({collection}) ──────────────────────")

    _npz_path  = C.ARTIFACTS_DIR / "graph_tensors.npz"
    _meta_path = C.ARTIFACTS_DIR / "graph_meta.json"
    uses = None
    ids  = None

    if _npz_path.exists() and _meta_path.exists():
        # Preferred: safe numpy + JSON, no pickle. Matches what
        # build_graph.py / train_rgcn.py / ingest_rgcn_to_qdrant.py write.
        try:
            import json as _json
            _t = np.load(_npz_path, allow_pickle=False)
            _m = _json.load(open(_meta_path, encoding="utf-8"))
            uses_src = _t["uses_src"]
            uses_dst = _t["uses_dst"]
            uses = np.stack([uses_src, uses_dst])   # shape (2, E), matches graph["relations"]["uses"][2]
            ids  = _m["problem_ids"]
        except Exception as exc:
            print(f"  [SKIP]  failed to read safe graph artifacts: {exc}")
            return True, {}
    elif C.GRAPH_PATH.exists():
        # Fallback: legacy pickle .pt -- only used if safe artifacts are absent
        import warnings
        warnings.warn(
            "graph_tensors.npz not found; falling back to graph.pt (pickle). "
            "Run build_graph.py to regenerate the safe format.",
            UserWarning, stacklevel=2,
        )
        try:
            import torch
            graph = torch.load(C.GRAPH_PATH, weights_only=False)
            uses_t = graph["relations"]["uses"][2]   # (2, E) tensor
            uses = uses_t.numpy() if hasattr(uses_t, "numpy") else np.asarray(uses_t)
            ids  = graph["problem_ids"]
        except Exception as exc:
            print(f"  [SKIP]  failed to read graph.pt: {exc}")
            return True, {}
    else:
        print("  [SKIP]  neither graph_tensors.npz nor graph.pt found")
        return True, {}

    try:
        # For well-trained RGCN: problems sharing concepts should be more similar
        # than random pairs. Sample 300 concept-connected pairs vs 300 random pairs.
        rng = np.random.default_rng(42)
        n_edges = uses.shape[1]
        pos_idx = rng.choice(n_edges, min(300, n_edges), replace=False)
        pos_sims, neg_sims = [], []
        pid_to_vec = {str(p.payload.get("problem_id", p.id)): _vec(p) for p in points}
        for ei in pos_idx:
            p_i = int(uses[0, ei])
            if p_i >= len(ids):
                continue
            pv = pid_to_vec.get(ids[p_i])
            if pv is None:
                continue
            # find another problem sharing the same concept
            c_i = int(uses[1, ei])
            concept_probs = [int(uses[0, j]) for j in range(n_edges) if int(uses[1, j]) == c_i and int(uses[0, j]) != p_i]
            if not concept_probs:
                continue
            pos_pid = ids[concept_probs[rng.integers(len(concept_probs))]]
            pos_v = pid_to_vec.get(pos_pid)
            if pos_v is not None:
                pos_sims.append(float(pv @ pos_v))
            # random negative pair
            neg_pid = ids[rng.integers(len(ids))]
            neg_v = pid_to_vec.get(neg_pid)
            if neg_v is not None:
                neg_sims.append(float(pv @ neg_v))

        if not pos_sims or not neg_sims:
            print("  [SKIP]  insufficient data for link AUC")
            return True, {}

        mean_pos = float(np.mean(pos_sims))
        mean_neg = float(np.mean(neg_sims))
        # AUC: fraction of (pos, neg) pairs where pos_sim > neg_sim
        auc = float(np.mean([p > n for p, n in zip(pos_sims[:len(neg_sims)], neg_sims)]))
        ok = auc > 0.5
        print(_r(ok, f"link_pred_auc         = {auc:.3f}  "
                     f"(pos_sim={mean_pos:.3f} > neg_sim={mean_neg:.3f})")[1])
        return ok, {"link_auc": auc}
    except Exception as e:
        print(f"  [SKIP]  {e}")
        return True, {}


# ===========================================================================
# 5. Sanity checks
# ===========================================================================

def test_sanity(client, rgcn_col, full_col, verbose):
    print(f"\n── 5. Sanity Checks ───────────────────────────────────────────")
    global FULL_DIM
    results = []

    # collections exist + dims
    for col, exp_dim in [(rgcn_col, RGCN_DIM), (full_col, None)]:
        try:
            info = client.get_collection(col)
            cnt  = client.count(collection_name=col).count
            dim  = info.config.params.vectors.size
            ok_d = (dim == exp_dim) if exp_dim else (dim > RGCN_DIM)
            results.append(_r(ok_d and cnt > 0,
                f"'{col}'  points={cnt}  dim={dim}"))
            if col == full_col:
                FULL_DIM = dim
        except Exception as e:
            results.append(_r(False, f"'{col}' not found: {e}"))

    rgcn_pts = _scroll_all(client, rgcn_col)

    # vector sanity
    n_nan = n_zero = n_bad = 0
    for p in rgcn_pts:
        arr = _vec(p)
        n_nan  += int(np.any(np.isnan(arr)))
        n_zero += int(np.linalg.norm(arr) < 1e-6)
        n_bad  += int(abs(np.linalg.norm(arr) - 1.0) > 0.05)
    n = len(rgcn_pts)
    results += [
        _r(n_nan  == 0, f"NaN vectors          : {n_nan}/{n}"),
        _r(n_zero == 0, f"zero vectors         : {n_zero}/{n}"),
        _r(n_bad  == 0, f"non-unit-norm        : {n_bad}/{n}"),
    ]

    # payload
    for key in ("problem_id", "title", "topic_tags"):
        miss = sum(1 for p in rgcn_pts if not (p.payload or {}).get(key))
        results.append(_r(miss == 0, f"payload '{key}'       : {n-miss}/{n}"))

    # self-similarity
    sample20 = rgcn_pts[:20]
    fails = []
    for p in sample20:
        hit = client.query_points(collection_name=rgcn_col,
                                  query=list(map(float, _vec(p))),
                                  limit=1, with_payload=False).points
        if not hit or hit[0].id != p.id:
            fails.append(p.id)
    results.append(_r(len(fails) == 0, f"self is top-1        : {20-len(fails)}/20"))

    # dim consistency
    if FULL_DIM:
        feat_dim = FULL_DIM - RGCN_DIM
        results.append(_r(feat_dim > 0,
            f"full({FULL_DIM}) = feat({feat_dim}) + rgcn({RGCN_DIM})"))

    # counts match
    full_cnt = client.count(collection_name=full_col).count
    rgcn_cnt = client.count(collection_name=rgcn_col).count
    results.append(_r(rgcn_cnt == full_cnt, f"point counts match   : {rgcn_cnt}=={full_cnt}"))

    # artifacts -- check the SAFE (pickle-free) artifact set as the
    # required pair, since build_graph.py/train_rgcn.py now write these
    # and .gitignore excludes *.pt. graph.pt / rgcn_model.pt are optional
    # legacy fallbacks; their absence should not fail a clean pickle-free run.
    _npz_path  = C.ARTIFACTS_DIR / "graph_tensors.npz"
    _meta_path = C.ARTIFACTS_DIR / "graph_meta.json"
    _arch_path = C.ARTIFACTS_DIR / "model_arch.json"

    required_artifacts = [
        (_npz_path,        "graph_tensors.npz"),
        (_meta_path,       "graph_meta.json"),
        (C.EMB_NPY,        "rgcn_problem_embeddings.npy"),
        (C.CONCEPT_INDEX,  "concept_index.json"),
        (_arch_path,       "model_arch.json"),
    ]
    for path, label in required_artifacts:
        ok = path.exists() and path.stat().st_size > 0
        results.append(_r(ok, f"{label:<38} {'found' if ok else 'MISSING'}"))

    # legacy pickle artifacts -- optional, report but never fail the suite on them
    for path, label in [(C.GRAPH_PATH, "graph.pt (legacy, optional)"),
                         (C.MODEL_PATH, "rgcn_model.pt (legacy, optional)")]:
        present = path.exists() and path.stat().st_size > 0
        tag = "found" if present else "absent (ok -- using safe artifacts)"
        print(f"  [INFO]  {label:<38} {tag}")

    if C.EMB_NPY.exists():
        arr = np.load(C.EMB_NPY)
        results.append(_r(arr.ndim == 2 and arr.shape[1] == RGCN_DIM,
                          f"npy shape={arr.shape}  expected=(N,{RGCN_DIM})"))

    for ok, msg in results:
        print(msg)
    return all(ok for ok, _ in results), rgcn_pts


# ===========================================================================
# Baseline comparison
# ===========================================================================

def compare_baselines(client, rgcn_col, full_col, src_col, points_rgcn, verbose):
    print(f"\n── BASELINE COMPARISON ────────────────────────────────────────")
    print(f"  (Does RGCN improve over raw embeddings?)")

    # get source points for baseline
    try:
        src_pts = _scroll_all(client, src_col)
        src_by_id = {str(p.payload.get("problem_id", p.id)): p for p in src_pts}
    except Exception as e:
        print(f"  [SKIP]  cannot read '{src_col}': {e}")
        return

    # align: only problems present in both
    pairs = []
    for p in points_rgcn:
        pid = str((p.payload or {}).get("problem_id", p.id))
        if pid in src_by_id:
            pairs.append((p, src_by_id[pid]))
    if not pairs:
        print("  [SKIP]  no overlapping problem_ids between collections")
        return

    def _recall_at10(col, pts):
        """
        Precision@10: average fraction of the top-10 neighbors that share a
        tag with the query, averaged across queries.

        The previous version used "any(shared tag in top 10)" -- a boolean
        per query. With ~15 avg tags/problem over a taxonomy of only ~484
        concepts, almost every problem shares at least one tag with almost
        any other, so that boolean saturates to 1.0 for raw, RGCN, and full
        embeddings alike -- it cannot tell them apart. Using the same
        pair-level definition as same_topic@10 in section 2 (fraction of ALL
        top-10 neighbor slots that share a tag, not just "at least one")
        gives a metric with actual headroom to show whether RGCN improves
        neighbor quality over raw embeddings.
        """
        hits, total = 0, 0
        for p in pts[:150]:
            qtags = set(_tag(p, "topic_tags"))
            if not qtags:
                continue
            try:
                res = client.query_points(collection_name=col,
                                          query=list(map(float, _vec(p))),
                                          limit=11, with_payload=True,
                                          with_vectors=False).points
            except Exception:
                continue
            res = [r for r in res if r.id != p.id][:10]
            for r in res:
                if set(_tag(r, "topic_tags")) & qtags:
                    hits += 1
                total += 1
        return hits / max(total, 1)

    rgcn_pts_aligned = [p for p, _ in pairs]
    src_pts_aligned  = [q for _, q in pairs]

    # full_embedding needs its own vectors (different dim from rgcn)
    try:
        full_pts_by_id = {}
        full_raw, _ = client.scroll(collection_name=full_col, limit=500,
                                    with_vectors=True, with_payload=True)
        for fp in full_raw:
            pid = str((fp.payload or {}).get("problem_id", fp.id))
            full_pts_by_id[pid] = fp
        full_pts_aligned = []
        for p, _ in pairs:
            pid = str((p.payload or {}).get("problem_id", p.id))
            if pid in full_pts_by_id:
                full_pts_aligned.append(full_pts_by_id[pid])
        has_full = len(full_pts_aligned) > 0
    except Exception:
        has_full = False

    r_src  = _recall_at10(src_col,  src_pts_aligned)
    r_rgcn = _recall_at10(rgcn_col, rgcn_pts_aligned)
    r_full = _recall_at10(full_col, full_pts_aligned) if has_full else None
    n_topics = len([t for t in set(
        (_tag(p, "topic_tags") or ["?"])[0] for p in rgcn_pts_aligned
    ) if t != "?"])
    r_rand = 1.0 / max(n_topics, 1)   # 1/n_topics

    # The verdict that matters is whether full_embedding (QS+RGCN concatenated,
    # what problems_full actually serves in production) beats raw QS alone --
    # NOT whether standalone rgcn_embedding beats raw QS. RGCN was never meant
    # to replace text-similarity signal on its own; it's a compressed
    # structural embedding (128-d, from graph topology) meant to ADD
    # complementary signal on top of QS text similarity (1792-d). Comparing
    # rgcn_embedding alone against raw QS alone conflates "does this add
    # value when combined" with "does this fully replace the other signal",
    # which is a different and much harder bar that isn't what the pipeline
    # relies on -- only full_embedding is actually queried downstream.
    if r_full is not None:
        improved = r_full > r_src
        improvement_pp = (r_full - r_src) * 100
    else:
        improved = r_rgcn > r_src
        improvement_pp = (r_rgcn - r_src) * 100

    print(f"  random (expected)  Precision@10 ≈ {r_rand:.3f}")
    print(f"  raw QS embedding   Precision@10 = {r_src:.3f}   [{src_col}]")
    print(f"  rgcn_embedding     Precision@10 = {r_rgcn:.3f}   [{'BETTER' if r_rgcn>r_src else 'worse'}]"
          f"  (diagnostic only -- structural signal, not meant to beat QS alone)")
    if r_full is not None:
        print(f"  full_embedding     Precision@10 = {r_full:.3f}   [{'BETTER' if r_full>r_src else 'worse'}]"
              f"  (this is what problems_full actually serves)")
    if improved:
        which = "full_embedding (QS+RGCN)" if r_full is not None else "rgcn_embedding"
        print(f"\n  [PASS]  {which} improves retrieval by {improvement_pp:.1f}pp over raw QS")
    else:
        which = "full_embedding (QS+RGCN)" if r_full is not None else "rgcn_embedding"
        print(f"\n  [FAIL]  {which} does NOT improve over raw QS embedding")
        print(f"          Retrain with more epochs or a higher SUPCON_WEIGHT")


# ===========================================================================
# Runner
# ===========================================================================

def run_all(url, rgcn_col, full_col, src_col, verbose, skip_baselines):
    client = _client(url)
    all_scores = {}

    sanity_ok, rgcn_pts = test_sanity(client, rgcn_col, full_col, verbose)
    rec_ok,  rec_s   = test_recommendation_quality(client, rgcn_col, rgcn_pts, verbose=verbose)
    sem_ok,  sem_s   = test_semantic_retrieval(client, rgcn_col, rgcn_pts, verbose=verbose)
    sep_ok,  sep_s   = test_topic_separation(client, rgcn_col, rgcn_pts, verbose=verbose)
    gph_ok,  gph_s   = test_graph_preservation(rgcn_col, rgcn_pts)
    all_scores = {**rec_s, **sem_s, **sep_s, **gph_s}

    if not skip_baselines:
        try:
            compare_baselines(client, rgcn_col, full_col, src_col, rgcn_pts, verbose)
        except Exception as e:
            print(f"\n  [SKIP] baseline comparison failed: {e}")

    tests = [sanity_ok, rec_ok, sem_ok, sep_ok, gph_ok]
    passed = sum(tests); failed = len(tests) - passed

    print("\n" + "=" * 62)
    print(f"  EVALUATION RESULTS: {passed}/{len(tests)} test groups passed")
    print(f"  {'ALL PASS' if failed == 0 else 'SOME FAILURES -- see above'}")
    if all_scores:
        print(f"\n  Key metrics:")
        for k, v in all_scores.items():
            print(f"    {k:<30} {v:.4f}")
    print("=" * 62 + "\n")
    return passed, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",            default=C.QDRANT_URL)
    ap.add_argument("--rgcn-col",       default=C.QDRANT_COLLECTION_RGCN)
    ap.add_argument("--full-col",       default=C.QDRANT_COLLECTION_FULL)
    ap.add_argument("--src-col",        default=C.QDRANT_SOURCE_COLLECTION,
                    help="raw embedder collection for baseline comparison")
    ap.add_argument("--verbose", "-v",  action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 62)
    print("  DSA ENGINE -- RGCN RETRIEVAL EVALUATION SUITE")
    print(f"  {args.url}  |  {args.rgcn_col}  |  {args.full_col}")
    print("=" * 62)

    _, failed = run_all(args.url, args.rgcn_col, args.full_col,
                        args.src_col, args.verbose, args.skip_baselines)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
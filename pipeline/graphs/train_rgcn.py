"""
DSA Engine -- RGCN Step 3: Train + Export
=========================================
Trains the HeteroRGCN encoder with a clean, balanced two-term objective:

    loss = SupCon(topic labels)  +  LINK_W * link_prediction(uses + cooccurs)

  * SupCon (Supervised Contrastive) is the PRIMARY signal -- it directly pulls
    same-topic problems together and pushes different-topic problems apart in
    one fully-vectorised term. This is what produces semantic clustering.
  * Link prediction on uses/cooccurs is a light auxiliary that keeps the
    concept-graph structure in the embedding (useful for the rec graph).

Why this replaces the previous version: the old triplet term was dwarfed by the
link losses (it stayed frozen at the margin floor and never trained), and three
competing link relations made the AUC drift down. SupCon is fully vectorised,
dominates the gradient, and its loss decreases monotonically alongside the
clustering metric.

Early stopping monitors the topic-clustering score directly (smooth now that
SupCon optimises it), with EMA damping.

Outputs: EMB_NPY (128-d), MODEL_PATH, optional parquet artifact.

Run:
    python rgcn/train_rgcn.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from rgcn_model import HeteroRGCN, LinkDecoder, precompute_adj, relation_types

# light auxiliary link-prediction relations (structure only; not the main signal)
SCORED   = ["uses", "cooccurs"]
LINK_W   = float(__import__("os").getenv("RGCN_LINK_WEIGHT", 0.1))   # light aux: don't let it drown SupCon
SUPCON_T = float(__import__("os").getenv("RGCN_SUPCON_TEMP", 0.5))   # warmer temp: safe for high-dim embeddings
SUPCON_W = float(__import__("os").getenv("RGCN_SUPCON_WEIGHT", 5.0)) # SupCon is the PRIMARY signal


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_edges(index, val_frac, seed):
    """
    Split edges into (train, val). Guarantees train is never empty:
    if a relation has only 1-2 edges, val gets 0 and everything goes to train
    rather than risking an empty positive batch -> NaN in BCE loss.
    """
    e = index.shape[1]
    if e == 0:
        empty = index[:, :0]
        return empty, empty
    perm = torch.randperm(e, generator=torch.Generator().manual_seed(seed))
    # only reserve a validation edge if there are enough edges that
    # training still has at least 1 left over
    n_val = min(max(1, int(e * val_frac)), e - 1) if e > 1 else 0
    return index[:, perm[n_val:]], index[:, perm[:n_val]]


def sample_neg(src, n_dst, k, pos_set, device):
    neg_dst = torch.randint(0, n_dst, (src.shape[0] * k,), device=device)
    rep_src = src.repeat_interleave(k)
    for _ in range(2):
        bad = torch.tensor(
            [(int(s), int(d)) in pos_set for s, d in zip(rep_src.tolist(), neg_dst.tolist())],
            device=device)
        if not bad.any():
            break
        neg_dst[bad] = torch.randint(0, n_dst, (int(bad.sum()),), device=device)
    return rep_src, neg_dst


# ---------------------------------------------------------------------------
# Supervised Contrastive loss (the clustering driver) -- fully vectorised
# ---------------------------------------------------------------------------

def supcon_loss(z, labels, temp=0.5):
    """
    Supervised Contrastive loss -- fully vectorised.
    z      : (n, d) L2-normalised problem embeddings
    labels : (n,)  integer topic label per row
    For each anchor, same-label rows are positives; everything else is negative.
    """
    n = z.shape[0]
    if n < 3:
        return z.new_tensor(0.0)
    z = F.normalize(z, dim=-1)
    # cosine similarity is in [-1, 1]; divide by temp AFTER clamping
    sim = torch.clamp(z @ z.T, -1.0, 1.0) / temp          # (n, n)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    # subtract max per row before exp for numerical stability (log-sum-exp trick)
    sim = sim - sim.detach().max(dim=1, keepdim=True).values
    sim = sim.masked_fill(self_mask, -1e4)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return z.new_tensor(0.0)
    log_prob = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True).clamp(min=1e-9))
    pos_log = (log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1).float()
    loss = -(pos_log[has_pos]).mean()
    return loss


@torch.no_grad()
def link_auc(decoder, z, rel, st, dt, val_index, n_dst, pos_set, device) -> float:
    pos = decoder(z[st], z[dt], val_index[0], val_index[1], rel)
    ns, nd = sample_neg(val_index[0], n_dst, 1, pos_set, device)
    neg = decoder(z[st], z[dt], ns, nd, rel)
    y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    s = torch.cat([pos, neg])
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, s.numel() + 1, device=device, dtype=torch.float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def cluster_score(zp, groups, device, n_topics=120) -> float:
    """Test-matching T5 metric: fraction of topics where a probe is closer to
    its own topic than to another. Fixed seed -> comparable across epochs."""
    topics = [t for t, idx in groups.items() if len(idx) >= 3]
    if len(topics) < 2:
        return 0.5
    zpn = F.normalize(zp, dim=-1)
    g = torch.Generator().manual_seed(123)
    if len(topics) > n_topics:
        topics = [topics[i] for i in torch.randperm(len(topics), generator=g)[:n_topics].tolist()]
    wins, trials = 0, 0
    for i, t in enumerate(topics):
        idx = groups[t]
        probe = zpn[idx[0]]
        same = zpn[idx[1:4]]
        diff = zpn[groups[topics[(i + 1) % len(topics)]][:3]]
        if same.shape[0] == 0 or diff.shape[0] == 0:
            continue
        wins += int(float((probe @ same.T).mean()) > float((probe @ diff.T).mean()))
        trials += 1
    return wins / max(trials, 1)


def _primary_topic_labels(graph):
    """Integer label per problem from topic_tags[0]; -1 if none."""
    cls_to_id, labels = {}, []
    for m in graph.get("problem_meta", []):
        tags = (m or {}).get("topic_tags") or []
        if tags:
            t = str(tags[0])
            labels.append(cls_to_id.setdefault(t, len(cls_to_id)))
        else:
            labels.append(-1)
    return torch.tensor(labels, dtype=torch.long), len(cls_to_id)


def _topic_groups(graph):
    groups = defaultdict(list)
    for i, m in enumerate(graph.get("problem_meta", [])):
        tags = (m or {}).get("topic_tags") or []
        if tags:
            groups[str(tags[0])].append(i)
    return {t: idx for t, idx in groups.items() if len(idx) >= 3}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train():
    print("\n" + "=" * 62)
    print("  RGCN STEP 3 -- TRAIN + EXPORT")
    print("=" * 62)
    seed_everything(C.SEED)
    device = torch.device(C.resolve_device())
    print(f"[->] device = {device}")

    # ── Load graph from safe npz + json (no pickle) ──────────────────────────
    _npz_path  = C.ARTIFACTS_DIR / "graph_tensors.npz"
    _meta_path = C.ARTIFACTS_DIR / "graph_meta.json"

    if _npz_path.exists() and _meta_path.exists():
        # Preferred path: safe numpy arrays + JSON metadata
        _t  = np.load(_npz_path, allow_pickle=False)
        _m  = json.load(open(_meta_path, encoding="utf-8"))
        uses_src    = torch.from_numpy(_t["uses_src"])
        uses_dst    = torch.from_numpy(_t["uses_dst"])
        uses_weight = torch.from_numpy(_t["uses_weight"])
        sim_index   = torch.from_numpy(_t["sim_index"])
        cc_index    = torch.from_numpy(_t["cc_index"])
        cc_arr      = _t["cc_weight"]
        cc_weight   = torch.from_numpy(cc_arr) if cc_arr.size else None
        X_prob      = torch.from_numpy(_t["problem_features"])
        X_con       = torch.from_numpy(_t["concept_features"])
        graph = {
            "node_features":     {"problem": X_prob, "concept": X_con},
            "num_nodes":         _m["num_nodes"],
            "relations": {
                "uses":    ("problem", "concept",
                            torch.stack([uses_src, uses_dst]), uses_weight),
                "used_by": ("concept", "problem",
                            torch.stack([uses_dst, uses_src]), uses_weight),
                "similar": ("problem", "problem", sim_index, None),
                "cooccurs":("concept", "concept", cc_index, cc_weight),
            },
            "problem_ids":      _m["problem_ids"],
            "problem_slugs":    _m["problem_slugs"],
            "problem_meta":     _m["problem_meta"],
            "problem_features": X_prob,
            "concepts":         _m["concepts"],
            "meta":             _m["meta"],
            "graph_fingerprint": _m.get("graph_fingerprint"),
        }
    else:
        # Fallback: legacy .pt — only safe if you built it yourself locally
        if not C.GRAPH_PATH.exists():
            raise FileNotFoundError(
                f"Neither {_npz_path} (safe) nor {C.GRAPH_PATH} (legacy) found -- "
                f"run build_graph.py first.")
        import warnings
        warnings.warn(
            "graph_tensors.npz not found; falling back to graph.pt (pickle). "
            "Run build_graph.py to regenerate the safe format.",
            UserWarning, stacklevel=2,
        )
        graph = torch.load(C.GRAPH_PATH, weights_only=False)  # nosec: local scratch only

    features = {t: v.to(device) for t, v in graph["node_features"].items()}
    adj = precompute_adj(graph, device)
    rels = relation_types(graph)
    in_dims = {t: features[t].shape[1] for t in features}

    model = HeteroRGCN(in_dims, rels, C.HIDDEN_DIM, C.OUT_DIM, C.DROPOUT).to(device)
    scored = [r for r in SCORED if graph["relations"][r][2].shape[1] > 0]
    decoder = LinkDecoder(C.OUT_DIM, scored).to(device)
    print(f"[OK] encoder params={sum(p.numel() for p in model.parameters()):,} "
          f"| link aux={scored} (w={LINK_W}) | SupCon temp={SUPCON_T} weight={SUPCON_W}")

    params = list(model.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(20, C.EARLY_STOP_PATIENCE // 4))

    rel_data = {}
    for r in scored:
        st, dt, ei, _ew = graph["relations"][r]
        train_e, val_e = split_edges(ei, C.VAL_FRACTION, C.SEED)
        pos_set = {(int(s), int(d)) for s, d in zip(ei[0].tolist(), ei[1].tolist())}
        rel_data[r] = (st, dt, train_e.to(device), val_e.to(device), pos_set,
                       graph["num_nodes"][dt])

    labels, n_cls = _primary_topic_labels(graph)
    labels = labels.to(device)
    labelled = (labels >= 0).nonzero(as_tuple=True)[0]      # rows with a topic
    groups = _topic_groups(graph)
    print(f"[OK] SupCon over {labelled.numel()} labelled problems, "
          f"{n_cls} topics | clustering monitor groups={len(groups)}")

    best, best_state, patience = -1.0, None, 0
    ema = None
    EMA_A = 0.2

    for epoch in range(1, C.EPOCHS + 1):
        model.train(); decoder.train()
        opt.zero_grad()
        z = model(features, adj)

        # ---- primary: supervised contrastive on topic labels ----
        sc = supcon_loss(z["problem"][labelled], labels[labelled], temp=SUPCON_T)

        # ---- auxiliary: light link prediction for graph structure ----
        link = z["problem"].new_tensor(0.0)
        n_scored = 0
        for r, (st, dt, tr, _v, pos_set, n_dst) in rel_data.items():
            if tr.shape[1] == 0:
                continue   # this relation's training split is empty -- skip
            pos = decoder(z[st], z[dt], tr[0], tr[1], r)
            ns, nd = sample_neg(tr[0], n_dst, C.NEG_PER_POS, pos_set, device)
            neg = decoder(z[st], z[dt], ns, nd, r)
            logits = torch.cat([pos, neg]); lab = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
            link = link + F.binary_cross_entropy_with_logits(logits, lab)
            n_scored += 1
        link = link / max(n_scored, 1)

        loss = SUPCON_W * sc + LINK_W * link

        # sanity: catch silent NaN (e.g. temp too cold causing exp overflow)
        if epoch == 1 and (torch.isnan(sc) or sc.item() > 50):
            print(f"  [WARN] supcon={sc.item():.3f} at epoch 1 -- possible NaN/overflow. "
                  f"Try RGCN_SUPCON_TEMP=1.0 if this doesn't decrease.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        # ---- validation: clustering (monitor) + link AUC (log) ----
        model.eval(); decoder.eval()
        with torch.no_grad():
            z = model(features, adj)
            clus = cluster_score(z["problem"], groups, device)
            aucs = [link_auc(decoder, z, r, st, dt, val_e, n_dst, ps, device)
                    for r, (st, dt, _t, val_e, ps, n_dst) in rel_data.items()]
            avg_auc = float(np.mean(aucs)) if aucs else 0.5

        ema = clus if ema is None else EMA_A * clus + (1 - EMA_A) * ema
        sched.step(ema)
        if ema > best:
            best = ema
            best_state = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()})
            patience = 0
        else:
            patience += 1

        if epoch % 20 == 0 or epoch == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"  epoch {epoch:>3}  loss={loss.item():.3f}  "
                  f"supcon={sc.item():.3f}  link={link.item():.3f}  "
                  f"auc={avg_auc:.4f}  cluster={clus:.3f}  ema={ema:.3f}  "
                  f"best={best:.3f}  lr={lr:.1e}")
        if patience >= C.EARLY_STOP_PATIENCE:
            print(f"  early stop @ epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state[0]); decoder.load_state_dict(best_state[1])
    print(f"[OK] best clustering (EMA) = {best:.3f}")

    model.eval()
    with torch.no_grad():
        z = model(features, adj)
        prob_emb = F.normalize(z["problem"], dim=-1).cpu().numpy().astype(np.float32)
        final_clus = cluster_score(torch.from_numpy(prob_emb), groups, torch.device("cpu"))
    print(f"[OK] final topic-clustering score = {final_clus:.3f}")

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.EMB_NPY, prob_emb)
    # Weights-only checkpoint: only state_dicts (tensors) — safe to load
    torch.save({
        "encoder": model.state_dict(),
        "decoder": decoder.state_dict(),
    }, C.MODEL_PATH)
    # Architecture metadata saved as JSON (no pickle, human-readable)
    # graph_fingerprint propagates from graph_meta.json so ingest_rgcn_to_qdrant.py
    # can detect a stale combination of npz/npy from different graph builds.
    _arch_path = C.ARTIFACTS_DIR / "model_arch.json"
    json.dump({
        "in_dims":      in_dims,
        "relations":    rels,
        "scored":       scored,
        "hidden":       C.HIDDEN_DIM,
        "out":          C.OUT_DIM,
        "best_cluster": best,
        "final_cluster": final_clus,
        "graph_fingerprint": graph.get("graph_fingerprint"),
    }, open(_arch_path, "w", encoding="utf-8"), indent=2)
    print(f"[OK] embeddings -> {C.EMB_NPY}  shape={prob_emb.shape}")
    print(f"[OK] model      -> {C.MODEL_PATH}  (weights only, safe)")
    print(f"[OK] arch meta  -> {_arch_path}")

    if C.WRITE_PARQUET:
        _write_parquet_artifact(graph, prob_emb)
    else:
        print("[->] WRITE_PARQUET=0 -- embeddings go to Qdrant in step 4")

    # in-process retrieval quality check (no Qdrant needed)
    _eval_retrieval_quality(graph, prob_emb)

    print("\n[OK] training complete")
    return final_clus

# ---------------------------------------------------------------------------
# Optional parquet artifact
# ---------------------------------------------------------------------------

def _eval_retrieval_quality(graph, prob_emb):
    """
    In-process retrieval check (no Qdrant).
    Answers: "Did RGCN embeddings improve same-topic retrieval over random?"
    Uses exact cosine KNN on the 128-d embeddings.
    """
    # FIX: the box-drawing characters here (U+2500 etc.) crashed the whole
    # pipeline with UnicodeEncodeError on Windows consoles using the
    # default cp1252 codepage -- this print happened right before the
    # Qdrant re-ingest step, so a purely cosmetic string crashed the run
    # AFTER training/embeddings/parquet had already succeeded, silently
    # skipping the re-ingest that was the actual point of the run. ASCII
    # only, matching every other status line in this pipeline ([OK]/[->]/[X]).
    print("\n-- In-process Retrieval Quality (same_topic@10) ----------------")
    meta = graph.get("problem_meta", [])
    # build topic groups
    by_topic = defaultdict(list)
    for i, m in enumerate(meta):
        tags = (m or {}).get("topic_tags") or []
        if tags:
            by_topic[str(tags[0])].append(i)
    topics_with_data = [t for t, idx in by_topic.items() if len(idx) >= 3]
    if len(topics_with_data) < 2:
        print("  [SKIP]  not enough topic data in meta")
        return

    emb = torch.from_numpy(prob_emb)   # (N, 128)
    emb = F.normalize(emb, dim=-1)
    n_queries, hits_rgcn, hits_rand = 0, 0, 0
    rng = np.random.default_rng(42)
    K = 10

    for t in topics_with_data[:30]:
        idx = by_topic[t]
        for probe_i in idx[:3]:
            probe = emb[probe_i]
            sims = (emb @ probe).numpy()
            sims[probe_i] = -2.0           # exclude self
            top_k = np.argpartition(sims, -K)[-K:]
            same_topic_ids = set(idx) - {probe_i}
            # RGCN hit: any top-K result in same topic
            hits_rgcn += int(bool(same_topic_ids & set(top_k.tolist())))
            # random baseline: expected hit rate = (|topic|-1) / (N-1)
            hits_rand += (len(idx) - 1) / max(len(emb) - 1, 1)
            n_queries += 1

    recall_rgcn = hits_rgcn / max(n_queries, 1)
    recall_rand = hits_rand / max(n_queries, 1)
    improved = recall_rgcn > recall_rand * 1.5   # at least 1.5x over random

    print(f"  random baseline   same_topic@{K} ~= {recall_rand:.3f}")
    ratio = recall_rgcn / max(recall_rand, 1e-4)
    tag = f"BETTER x{ratio:.1f}" if recall_rgcn > recall_rand else "not better than random"
    print(f"  rgcn_embedding    same_topic@{K}  = {recall_rgcn:.3f}  [{tag}]")
    if improved:
        print(f"  [PASS]  retrieval improved over random")
    else:
        print(f"  [WARN]  retrieval not clearly above random -- consider more epochs or SUPCON_WEIGHT")


def _write_parquet_artifact(graph, prob_emb) -> None:
    import pandas as pd
    ids = graph["problem_ids"]
    feats = graph["problem_features"].numpy().astype(np.float32)
    full = []
    for i in range(len(ids)):
        v = np.concatenate([feats[i], prob_emb[i]]).astype(np.float32)
        n = np.linalg.norm(v)
        full.append((v / (n or 1.0)).astype(np.float32))

    if C.INPUT_PARQUET.exists():
        df = pd.read_parquet(C.INPUT_PARQUET)
        id_to_r = {pid: prob_emb[i] for i, pid in enumerate(ids)}
        id_to_f = {pid: full[i] for i, pid in enumerate(ids)}
        df["rgcn_embedding"] = pd.Series([id_to_r.get(str(p)) for p in df["problem_id"]], dtype=object)
        df["full_embedding"] = pd.Series([id_to_f.get(str(p)) for p in df["problem_id"]], dtype=object)
    else:
        meta = graph.get("problem_meta", [{} for _ in ids])
        df = pd.DataFrame({"problem_id": ids,
                           "rgcn_embedding": pd.Series(list(prob_emb), dtype=object),
                           "full_embedding": pd.Series(full, dtype=object)})
        for k in ("title", "title_slug", "difficulty_score"):
            df[k] = [m.get(k) for m in meta]

    C.OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.OUTPUT_PARQUET, index=False)
    print(f"[OK] parquet artifact -> {C.OUTPUT_PARQUET}  "
          f"(rgcn dim {prob_emb.shape[1]}, full dim {len(full[0])})")


if __name__ == "__main__":
    train()
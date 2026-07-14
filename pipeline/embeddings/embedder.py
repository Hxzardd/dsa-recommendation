"""
DSA Engine -- Embedding Generator
==================================
Generates semantic embeddings for the vector pool.

Pipeline per problem:
    Question  : title + explanation_text + tag phrases  -> BGE-Large    -> Q (1024-dim)
    Solution  : Tree-sitter -> AST -> CFG + DFG -> PDG  -> GraphCodeBERT -> S (768-dim)
    Combined  : concat(Q, S) re-normalised              -> QS (1792-dim)

Checkpointing: saves progress every CHECKPOINT_EVERY rows so a crash
never loses more than that many rows of work. Use --resume to continue.

Usage:
    python embedder.py --input vector_pool/vector_pool.parquet --output vector_pool/
    python embedder.py --input vector_pool/vector_pool.parquet --output vector_pool/ --resume
    python embedder.py --input vector_pool/vector_pool.parquet --output vector_pool/ --batch-size 8
    python embedder.py --input vector_pool/vector_pool.parquet --output vector_pool/ \
        --qdrant-url http://localhost:6333 --collection dsa_problems

Dependencies:
    uv pip install sentence-transformers torch numpy qdrant-client tree-sitter tree-sitter-python
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# This file lives at pipeline/embeddings/ -- walk up to the repo root to
# find db_env.py, the one canonical .env-loading module. Only needed for
# the --qdrant-api-key default below; embedder.py still works fine if
# db_env.py can't be found (e.g. run completely standalone outside the
# repo) -- it just falls back to no default API key, same as before.
try:
    _here = Path(__file__).resolve()
    for _p in [_here.parent, *_here.parents]:
        if (_p / "pyproject.toml").exists():
            sys.path.insert(0, str(_p))
            break
    import db_env
    _DEFAULT_QDRANT_API_KEY = db_env.QDRANT_API_KEY
except ImportError:
    _DEFAULT_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

import numpy as np
import pandas as pd

# How often to write a checkpoint parquet during embedding
CHECKPOINT_EVERY = 100


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

_BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def build_question_text(row: dict) -> str:
    """
    Builds input for BGE-Large:
        title | explanation_text | Topics: ... Algorithms: ... Patterns: ...
    """
    parts = []

    def _str(val):
        if val is None: return ""
        try: return str(val).strip()
        except: return ""

    title = _str(row.get("title"))
    if title:
        parts.append(title)

    exp = _str(row.get("explanation_text"))
    if exp and exp != 'nan':
        parts.append(exp[:800])

    tag_phrases = []

    def _to_list(val):
        # pandas reads list columns from parquet as numpy arrays
        if val is None:
            return []
        try:
            return list(val)
        except Exception:
            return []

    topics   = _to_list(row.get("topic_tags"))
    algos    = _to_list(row.get("algorithm_tags"))
    patterns = _to_list(row.get("patterns"))
    if topics:   tag_phrases.append("Topics: "     + ", ".join(str(t).replace("_", " ") for t in topics))
    if algos:    tag_phrases.append("Algorithms: " + ", ".join(str(t).replace("_", " ") for t in algos))
    if patterns: tag_phrases.append("Patterns: "   + ", ".join(str(t).replace("_", " ") for t in patterns))
    if tag_phrases:
        parts.append(". ".join(tag_phrases))

    return " | ".join(parts)


def build_solution_text(row: dict) -> Optional[str]:
    """
    Builds semantically-enriched input for GraphCodeBERT.
    Uses AST -> CFG + DFG -> PDG analysis via code_analyzer.py.
    Falls back to raw code if tree-sitter is unavailable.
    """
    sol = row.get("canonical_solution")
    # parquet may return numpy scalar — convert to str safely
    if sol is None or (hasattr(sol, '__len__') and len(sol) == 0):
        return None
    try:
        sol = str(sol).strip()
    except Exception:
        return None
    if not sol or sol == 'nan':
        return None

    try:
        from utils.code_analyzer import build_semantic_text
        enriched = build_semantic_text(sol)
        if enriched:
            return enriched[:1500]
    except Exception:
        pass

    # Fallback: raw code
    return sol[:1500]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

_question_model = None
_solution_model = None


def _load_question_model():
    global _question_model
    if _question_model is None:
        from sentence_transformers import SentenceTransformer
        print("[->] Loading BAAI/bge-large-en-v1.5 (question encoder)...")
        _question_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        print("[OK] Question model loaded")
    return _question_model


def _load_solution_model():
    global _solution_model
    if _solution_model is None:
        from sentence_transformers import SentenceTransformer
        print("[->] Loading microsoft/graphcodebert-base (solution encoder)...")
        _solution_model = SentenceTransformer("microsoft/graphcodebert-base")
        print("[OK] Solution model loaded")
    return _solution_model


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def embed_questions(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = _load_question_model()
    prefixed = [_BGE_INSTRUCTION + t for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def embed_solutions(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = _load_solution_model()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def concat_and_normalise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Concat two embedding arrays row-wise and L2-normalise."""
    combined = np.concatenate([a, b], axis=1).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return combined / norms


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir) / "vector_pool_checkpoint.parquet"


def save_checkpoint(df: pd.DataFrame, output_dir: str) -> None:
    path = _checkpoint_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    n_done = df["question_embedding"].notna().sum()
    print(f"  [checkpoint] {n_done}/{len(df)} rows saved -> {path}")


def load_checkpoint(output_dir: str) -> Optional[pd.DataFrame]:
    path = _checkpoint_path(output_dir)
    if path.exists():
        df = pd.read_parquet(path)
        n_done = df["question_embedding"].notna().sum()
        print(f"[->] Checkpoint found: {n_done}/{len(df)} rows already embedded")
        return df
    return None


# ---------------------------------------------------------------------------
# Main embedding loop
# ---------------------------------------------------------------------------

def embed_dataframe(
    df: pd.DataFrame,
    batch_size: int = 32,
    resume: bool = False,
    output_dir: str = "./vector_pool",
) -> pd.DataFrame:
    """
    Generate embeddings for all rows.
    Checkpoints every CHECKPOINT_EVERY rows so crashes lose minimal work.
    """
    df = df.copy()
    n  = len(df)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if resume:
        ckpt = load_checkpoint(output_dir)
        if ckpt is not None:
            # Merge checkpoint embeddings into df
            embed_cols = [
                "question_embedding", "solution_embedding",
                "rgcn_embedding", "question_solution_embedding", "full_embedding",
            ]
            for col in embed_cols:
                if col in ckpt.columns:
                    df[col] = ckpt[col].values
        # Resume: a row needs work if ANY of the three embedding columns is missing.
        # Checking only question_embedding would silently skip solution/combined
        # if the process crashed after step 1 (Greptile P1 bug).
        todo_mask = (
            df["question_embedding"].isna()
            | df["solution_embedding"].isna()
            | df["question_solution_embedding"].isna()
        )
        todo_idx  = df.index[todo_mask].tolist()
        done_n    = n - len(todo_idx)
        if done_n > 0:
            print(f"[->] Resuming: {done_n} rows done, {len(todo_idx)} remaining")
    else:
        todo_idx = df.index.tolist()

    if not todo_idx:
        print("[OK] All rows already embedded -- nothing to do")
        return df

    sub = df.loc[todo_idx]
    print(f"[->] Embedding {len(sub)} rows (batch_size={batch_size})")

    # ── Step 1: Question embeddings (BGE-Large) ──────────────────────────────
    print(f"\n[1/3] Building question texts...")
    q_texts = [build_question_text(row) for _, row in sub.iterrows()]

    print("[1/3] Encoding question embeddings (BGE-Large)...")
    t0     = time.time()
    q_embs = embed_questions(q_texts, batch_size=batch_size)
    q_dim  = q_embs.shape[1]
    print(f"[OK] Questions done in {time.time()-t0:.1f}s  shape={q_embs.shape}")

    # Write Q embeddings to df immediately -- safe even if step 2 crashes
    for df_row_i, local_i in enumerate(todo_idx):
        df.at[local_i, "question_embedding"] = q_embs[df_row_i]
    save_checkpoint(df, output_dir)

    # ── Step 2: Solution embeddings (GraphCodeBERT + semantic analysis) ───────
    print(f"\n[2/3] Building semantic solution texts (AST->CFG/DFG->PDG)...")

    # Check if tree-sitter is available
    try:
        from utils.code_analyzer import build_semantic_text
        print("[OK] Tree-sitter available -- using AST/CFG/DFG/PDG enrichment")
        semantic_mode = True
    except ImportError:
        print("[!]  tree-sitter not installed -- falling back to raw code")
        print("     Install with: uv pip install tree-sitter tree-sitter-python")
        semantic_mode = False

    sol_texts_raw = [build_solution_text(row) for _, row in sub.iterrows()]
    has_sol_mask  = [t is not None for t in sol_texts_raw]
    sol_texts     = [t for t in sol_texts_raw if t is not None]

    s_dim = 768  # GraphCodeBERT always 768
    sol_embs_full = np.full((len(sub), s_dim), np.nan, dtype=np.float32)

    if sol_texts:
        print(f"[2/3] Encoding {len(sol_texts)} solution embeddings (GraphCodeBERT)...")
        t1       = time.time()
        sol_embs = embed_solutions(sol_texts, batch_size=batch_size)
        s_dim    = sol_embs.shape[1]
        sol_embs_full = np.full((len(sub), s_dim), np.nan, dtype=np.float32)
        print(f"[OK] Solutions done in {time.time()-t1:.1f}s  shape={sol_embs.shape}")

        sol_idx = 0
        for i, has in enumerate(has_sol_mask):
            if has:
                sol_embs_full[i] = sol_embs[sol_idx]
                sol_idx += 1
    else:
        print("[!] No solution texts to embed")

    # Write S embeddings to df
    for df_row_i, local_i in enumerate(todo_idx):
        s = sol_embs_full[df_row_i]
        df.at[local_i, "solution_embedding"] = None if np.any(np.isnan(s)) else s
    save_checkpoint(df, output_dir)

    # ── Step 3: Concat Q + S ─────────────────────────────────────────────────
    qs_dim = q_dim + s_dim
    print(f"\n[3/3] Building concat embeddings (dim={q_dim}+{s_dim}={qs_dim})...")

    qs_embs_full = np.full((len(sub), qs_dim), np.nan, dtype=np.float32)
    valid_both   = [i for i, has in enumerate(has_sol_mask) if has]

    if valid_both:
        q_sub  = q_embs[valid_both]
        s_sub  = sol_embs_full[valid_both]
        qs_sub = concat_and_normalise(q_sub, s_sub)
        for out_i, in_i in enumerate(valid_both):
            qs_embs_full[in_i] = qs_sub[out_i]

    print(f"[OK] Concat done  shape=(N, {qs_dim})")

    # Write QS embeddings + remaining cols to df
    for df_row_i, local_i in enumerate(todo_idx):
        qs = qs_embs_full[df_row_i]
        df.at[local_i, "question_solution_embedding"] = (
            None if np.any(np.isnan(qs)) else qs
        )
        df.at[local_i, "rgcn_embedding"] = None
        df.at[local_i, "full_embedding"] = None

    # Final checkpoint
    save_checkpoint(df, output_dir)
    print("[OK] DataFrame updated")
    return df


# ---------------------------------------------------------------------------
# Save final output
# ---------------------------------------------------------------------------

def save_embedded(df: pd.DataFrame, output_dir: str) -> None:
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "vector_pool_embedded.parquet"
    df.to_parquet(path, index=False)
    print(f"[OK] Saved -> {path}  ({len(df)} rows)")

    n_q  = df["question_embedding"].notna().sum()
    n_s  = df["solution_embedding"].notna().sum()
    n_qs = df["question_solution_embedding"].notna().sum()

    q_sample  = df["question_embedding"].dropna()
    qs_sample = df["question_solution_embedding"].dropna()
    q_dim     = len(q_sample.iloc[0])  if len(q_sample)  > 0 else "?"
    qs_dim    = len(qs_sample.iloc[0]) if len(qs_sample) > 0 else "?"

    print(f"    question_embedding          : {n_q}/{len(df)}  dim={q_dim}")
    print(f"    solution_embedding          : {n_s}/{len(df)}  dim=768")
    print(f"    question_solution_embedding : {n_qs}/{len(df)}  dim={qs_dim}")
    print(f"    rgcn_embedding              : 0/{len(df)} (future -- needs RGCN training)")
    print(f"    full_embedding              : 0/{len(df)} (future -- needs rgcn_embedding)")

    # Clean up checkpoint file after successful save
    ckpt = _checkpoint_path(output_dir)
    if ckpt.exists():
        ckpt.unlink()
        print(f"[OK] Checkpoint removed (no longer needed)")


# ---------------------------------------------------------------------------
# Qdrant upload
# ---------------------------------------------------------------------------


_RETRYABLE = (
    "WriteTimeout", "ReadTimeout", "ConnectTimeout",
    "TimeoutException", "ConnectionError", "RemoteProtocolError",
    "WriteError", "ReadError",
)

def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    msg  = str(exc).lower()
    if any(r in name for r in _RETRYABLE):
        return True
    if "responsehandlingexception" in name.lower():
        return True
    return False

def _upsert_with_retry(client, collection: str, batch: list, max_retries: int = 4) -> None:
    """
    Retry a single batch upsert on TRANSIENT network errors only.
    Permanent errors (403, dimension mismatch, missing collection)
    are re-raised immediately -- fail fast, no retry.
    Upsert is idempotent by point ID so retrying is always safe.
    """
    import time as _time
    last_exc = None
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection, points=batch)
            return
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n[!] Transient upload error ({exc.__class__.__name__}), "
                      f"retrying in {wait}s (attempt {attempt + 2}/{max_retries})...")
                _time.sleep(wait)
    raise last_exc



def upload_to_qdrant(
    df: pd.DataFrame,
    url: str,
    collection: str,
    embedding_col: str = "question_solution_embedding",
    batch_size: int = 50,
    api_key: str = None,
    upload_timeout: int = 120,
    max_retries: int = 4,
) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct, OptimizersConfigDiff,
        )
    except ImportError:
        print("[X] qdrant-client not installed. Run: uv pip install qdrant-client")
        return

    # api_key=None is fine for a local Docker instance (no auth). For a
    # cloud Qdrant instance (Qdrant Cloud, etc.) this must be set -- passed
    # explicitly here rather than assumed, so a local run and a cloud run
    # both work with the same code path.
    #
    # Two different timeouts on purpose: a SHORT one (5s) just to fail
    # fast if Qdrant is completely unreachable, and a much LONGER one
    # (upload_timeout, default 120s) for the actual client used to upload
    # data. The two are unrelated operations with very different latency
    # needs -- a single 5s timeout reused for both was the actual cause of
    # WriteTimeout crashes partway through large batch uploads to a cloud
    # instance: get_collections() is a tiny call that returns in
    # milliseconds even with a generous timeout ceiling, but upserting a
    # 50-100 point batch (each up to 1920 floats) over a real network
    # connection to a cloud region genuinely needs more than 5 seconds,
    # especially under any network variance.
    try:
        probe = QdrantClient(url=url, api_key=api_key, timeout=5)
        probe.get_collections()   # fast connectivity check only
    except Exception as e:
        print(f'[X] Cannot connect to Qdrant at {url}')
        print(f'    Error: {e.__class__.__name__}: {str(e)[:120]}')
        print()
        if api_key is None:
            print('    No API key was provided. If this is a cloud Qdrant')
            print('    instance, set QDRANT_API_KEY in your .env or pass')
            print('    --qdrant-api-key explicitly.')
        print('    If this is meant to be a local instance, start it with:')
        print('      docker pull qdrant/qdrant')
        print('      docker run -p 6333:6333 qdrant/qdrant')
        print()
        print('    Then re-run the upload command (embeddings are already saved):')
        print(f'      uv run embedder.py --input vector_pool/vector_pool_embedded.parquet --output vector_pool --qdrant-url {url} --collection {collection}')
        return

    # Separate client for the actual work, with the generous timeout --
    # the probe above already confirmed Qdrant is reachable at all.
    client = QdrantClient(url=url, api_key=api_key, timeout=upload_timeout)

    sample_col = df[embedding_col].dropna()
    if len(sample_col) == 0:
        embedding_col = "question_embedding"
        sample_col    = df[embedding_col].dropna()

    vec_size = len(sample_col.iloc[0])
    print(f"[->] Qdrant at {url}  |  collection={collection}  |  dim={vec_size}")

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        print(f"[OK] Collection '{collection}' created")
    else:
        existing_dim = client.get_collection(collection).config.params.vectors.size
        if existing_dim != vec_size:
            raise ValueError(
                f"Collection '{collection}' exists with dim={existing_dim} but "
                f"you're uploading --qdrant-vector={embedding_col} which is "
                f"dim={vec_size}. Use a different --collection name per vector "
                f"type, or delete the existing collection first. "
                f"Refusing to upsert mismatched dimensions.")
        print(f"[->] Collection '{collection}' exists -- upserting "
              f"(dim={vec_size} confirmed match)")

    points  = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        vec = row.get(embedding_col)
        # No per-row fallback to a different column here. The collection was
        # created with a FIXED dimension (vec_size, from embedding_col above).
        # question_embedding is 1024-d while question_solution_embedding is
        # 1792-d -- silently substituting one for the other per-row sends a
        # wrong-dimension vector into a fixed-dimension collection and Qdrant
        # rejects the whole batch with a 400. Rows missing embedding_col are
        # skipped entirely instead; they can be uploaded separately into a
        # different collection sized for question_embedding if needed.
        if vec is None:
            skipped += 1
            continue

        payload = {
            col: (row[col].tolist() if isinstance(row[col], np.ndarray) else row[col])
            for col in df.columns
            if col not in (
                "question_embedding", "solution_embedding", "rgcn_embedding",
                "question_solution_embedding", "full_embedding",
            )
            and not (isinstance(row[col], float) and np.isnan(row[col]))
        }
        points.append(PointStruct(id=i, vector=vec.tolist(), payload=payload))

    print(f"[->] Uploading {len(points)} points ({skipped} skipped)...")
    t0 = time.time()
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        _upsert_with_retry(client, collection, batch, max_retries=max_retries)
        pct = min(100, int(100 * (start + len(batch)) / len(points)))
        print(f"\r  {pct}% ({start + len(batch)}/{len(points)})", end="", flush=True)

    client.update_collection(
        collection_name=collection,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
    )
    print(f"\n[OK] Upload done in {time.time()-t0:.1f}s")
    print(f"[OK] Collection '{collection}' has {client.count(collection_name=collection).count} points")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DSA Engine -- Embedding Generator")
    p.add_argument("--input",      "-i", default="./data/vector_pool/vector_pool.parquet")
    p.add_argument("--output",     "-o", default="./data/vector_pool")
    p.add_argument("--batch-size", "-b", type=int, default=32,
                   help="Rows per model forward pass (lower to 8 if RAM is tight)")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from checkpoint -- skips already-embedded rows")
    p.add_argument("--qdrant-url",    default=None,
                   help="Qdrant URL e.g. http://localhost:6333 or a cloud "
                        "Qdrant URL (optional)")
    p.add_argument("--qdrant-api-key", default=_DEFAULT_QDRANT_API_KEY,
                   help="Qdrant API key -- required for cloud Qdrant, "
                        "unnecessary for a local unauthenticated instance. "
                        "Defaults to the QDRANT_API_KEY environment variable.")
    p.add_argument("--collection",    default="dsa_problems")
    p.add_argument("--qdrant-upload-batch-size", type=int, default=50,
                   help="Points per Qdrant upsert call (lower this if you "
                        "still see WriteTimeout errors on a slow connection)")
    p.add_argument("--qdrant-upload-timeout", type=int, default=120,
                   help="Seconds to wait per upsert call before timing out "
                        "(raise this for large batches over a cloud connection)")
    p.add_argument("--qdrant-max-retries", type=int, default=4,
                   help="Retry attempts per batch on transient network errors, "
                        "with exponential backoff")
    p.add_argument("--qdrant-vector", default="question_solution_embedding",
                   choices=["question_embedding", "solution_embedding",
                            "question_solution_embedding"])
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*62)
    print("  DSA ENGINE -- EMBEDDING GENERATOR")
    print("="*62)

    path = Path(args.input)
    if not path.exists():
        print(f"[X] File not found: {path}")
        print("    Run run_pipeline.py first to generate vector_pool.parquet")
        sys.exit(1)

    print(f"[->] Loading {path}...")
    df = pd.read_parquet(path)
    print(f"[OK] {len(df)} rows x {len(df.columns)} columns loaded")

    # Skip re-embedding if all rows already have question_embedding populated
    already_done = df['question_embedding'].notna().all()
    if already_done and not args.resume:
        print('[OK] All embeddings already present -- skipping embedding step')
        n_q  = df['question_embedding'].notna().sum()
        n_s  = df['solution_embedding'].notna().sum()
        n_qs = df['question_solution_embedding'].notna().sum()
        print(f'    question_embedding          : {n_q}/{len(df)}')
        print(f'    solution_embedding          : {n_s}/{len(df)}')
        print(f'    question_solution_embedding : {n_qs}/{len(df)}')
        # Save embedded parquet if not already the embedded file
        embedded_path = Path(args.output) / "vector_pool_embedded.parquet"
        if path.resolve() != embedded_path.resolve():
            save_embedded(df, args.output)
    else:
        df = embed_dataframe(
            df,
            batch_size=args.batch_size,
            resume=args.resume,
            output_dir=args.output,
        )
        save_embedded(df, args.output)

    if args.qdrant_url:
        print("\n[->] Uploading to Qdrant...")
        upload_to_qdrant(
            df,
            url=args.qdrant_url,
            collection=args.collection,
            embedding_col=args.qdrant_vector,
            api_key=args.qdrant_api_key,
            batch_size=args.qdrant_upload_batch_size,
            upload_timeout=args.qdrant_upload_timeout,
            max_retries=args.qdrant_max_retries,
        )

    print("\n" + "="*62)
    print("  EMBEDDING COMPLETE")
    print(f"  Output: {(Path(args.output) / 'vector_pool_embedded.parquet').resolve()}")
    if args.qdrant_url:
        print(f"  Qdrant: {args.qdrant_url} / {args.collection}")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
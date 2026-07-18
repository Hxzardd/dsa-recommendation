"""
DSA Engine -- Vector Pool Ingestion
=====================================
Transforms raw manifest records into the canonical vector pool schema.

Exported API:
    build_vector_pool(records)  -> pd.DataFrame
    save_outputs(df, output_dir)
    preview(df, n)
    load_manifest(path)         -> list
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from schema import FINAL_COLUMNS, EXCLUDE_COLUMNS
from tag_inference import (
    infer_difficulty_score,
    infer_solution_signature,
    pick_canonical_solution,
    extract_similar_problem_ids,
)


# ---------------------------------------------------------------------------
# Canonical topic map -- built once at module load from 
# source-target.txt. Maps title_slug -> [canonical_tag, ...]. If the file
# doesn't exist yet (first run before generate_topic_edges), falls back to
# an empty dict so ingestion still works, just with AI-inferred tags.
# ---------------------------------------------------------------------------

def _build_canonical_topic_map() -> dict:
    _here = Path(__file__).resolve()
    for _p in [_here.parent, *_here.parents]:
        if (_p / "pyproject.toml").exists():
            _source_target = _p / "data" / "source-target.txt"
            if _source_target.exists():
                raw = json.load(open(_source_target, encoding="utf-8"))
                result = defaultdict(list)
                for e in raw:
                    src = (e.get("source") or "").strip()
                    tgt = (e.get("target") or "").strip()
                    if src and tgt:
                        result[src].append(tgt)
                print(f"[ingest] Loaded canonical topic map: "
                      f"{len(result)} problems from source-target.txt")
                return dict(result)
            else:
                print(f"[ingest] source-target.txt not found at {_source_target} "
                      f"-- topic_tags will use AI-inferred values")
                return {}
    return {}

_CANONICAL_TOPIC_MAP: dict = _build_canonical_topic_map()


# ---------------------------------------------------------------------------
# Description builder
# ---------------------------------------------------------------------------

def _clean_markdown(text: str) -> str:
    """Strip markdown/LaTeX noise to produce clean embeddable prose."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)   # code fences
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL) # block LaTeX
    text = re.sub(r"\$.*?\$", "", text)                       # inline LaTeX
    text = re.sub(r"#{1,6}\s*", "", text)                     # headers
    text = re.sub(r"\[TOC\]", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)               # images
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)                # links
    text = re.sub(r"\*{1,3}", "", text)                       # bold/italic
    text = re.sub(r"`[^`]+`", "", text)                       # inline code
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_description(record: Dict[str, Any]) -> str:
    """
    Build the embeddable description field.
    Priority: explanation_text -> first 1000 chars of full_markdown_solution -> title.
    """
    exp = (record.get("explanation_text") or "").strip()
    if len(exp) > 40:
        return _clean_markdown(exp)

    md = (record.get("full_markdown_solution") or "").strip()
    if len(md) > 40:
        return _clean_markdown(md[:1000])

    return record.get("title", "Unknown Problem")


# ---------------------------------------------------------------------------
# Per-record transform
# ---------------------------------------------------------------------------

def transform_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw manifest dict into the canonical vector pool row.
    Column order matches FINAL_COLUMNS exactly.
    """
    out: Dict[str, Any] = {}

    # -- Identity -------------------------------------------------------------
    out["problem_id"]   = record.get("problem_id", "")
    out["title"]        = record.get("title", "")
    out["title_slug"]   = record.get("title_slug", "") or ""

    # -- Content --------------------------------------------------------------
    out["description"]         = build_description(record)
    out["explanation_text"]    = (record.get("explanation_text") or "").strip()
    out["solution_signature"]  = infer_solution_signature(record)
    out["canonical_solution"]  = pick_canonical_solution(record)

    # -- Signals ---------------------------------------------------------------
    out["difficulty_score"] = infer_difficulty_score(record)
    out["frequency"]        = record.get("frequency")       # None if missing
    out["rating"]           = record.get("rating")          # None if missing
    out["likes"]            = record.get("likes", 0) or 0
    out["dislikes"]         = record.get("dislikes", 0) or 0
    out["asked_by_faang"]   = bool(record.get("asked_by_faang", False))

    # -- Tags -------------------------------------------------------------------
    # FIX: this used to call infer_topic_tags(record) (tag_inference.py's
    # keyword/substring-matching heuristic against solution text -- an
    # AI/heuristic guess, not real data) as the default, only overriding it
    # with the reconciled canonical mapping when title_slug happened to
    # have an entry in source-target.txt -- and even then, silently kept
    # the AI-inferred guess as a "fallback" for any problem WITHOUT a
    # canonical entry (per this function's own prior comment: "we keep
    # whatever was inferred... so the field is never empty"). Confirmed
    # this left ~1755 of 2913 manifest problems (everything outside
    # source-target.txt's coverage) with fabricated topic_tags actually
    # live in Qdrant's payload. topic_tags is now ONLY ever the reconciled
    # canonical mapping, or [] if this problem has no entry there --
    # no guessing.
    #
    # algorithm_tags/data_structure_tags/patterns/techniques/skill_tags had
    # NO canonical override at all (100% tag_inference.py heuristic output,
    # confirmed unused anywhere in pipeline/recommender/'s actual
    # recommendation logic -- only topic_tags and difficulty_score are ever
    # read from a problem's payload). Hard-set to [] rather than removing
    # the columns outright, so downstream schema/parquet code that expects
    # these keys to exist doesn't need to change.
    slug = record.get("title_slug") or ""
    out["topic_tags"]          = _CANONICAL_TOPIC_MAP.get(slug, [])
    out["algorithm_tags"]      = []
    out["data_structure_tags"] = []
    out["patterns"]            = []
    out["techniques"]          = []
    out["skill_tags"]          = []

    # -- Relational ------------------------------------------------------------
    out["similar_problem_ids"] = extract_similar_problem_ids(record)
    out["companies"]           = record.get("companies") or []

    # -- Embeddings (all None -- populated by future pipelines) ----------------
    out["question_embedding"]           = None  # -> BGE Large
    out["solution_embedding"]           = None  # -> GraphCodeBERT
    out["rgcn_embedding"]               = None  # -> RGCN (future)
    out["question_solution_embedding"]  = None  # -> concat(Q, S)
    out["full_embedding"]               = None  # -> concat(Q, S, RGCN)

    return out


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Dict[str, Any]]:
    """Load manifest JSON from disk. Handles both list and wrapped-dict formats."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("problems", "data", "records", "questions"):
        if key in data:
            return data[key]
    raise ValueError(f"Cannot parse manifest -- top-level keys: {list(data.keys())}")


def build_vector_pool(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transform all raw records into the canonical DataFrame.
    Column order is exactly FINAL_COLUMNS from schema.py.
    Any column in EXCLUDE_COLUMNS that somehow leaked through is dropped.
    """
    rows = [transform_record(r) for r in records]
    df = pd.DataFrame(rows, columns=FINAL_COLUMNS)

    # Safety net -- drop any excluded columns that leaked
    leaked = set(df.columns) & EXCLUDE_COLUMNS
    if leaked:
        df = df.drop(columns=list(leaked))

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _to_json_safe(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to JSON-safe list, handling None, lists, and
    numpy arrays (embedding columns) correctly.

    pd.isna() on a numpy array returns an array of bools, not a scalar,
    which makes `if ... pd.isna(val)` raise "truth value of an array is
    ambiguous". Embedding columns (question_embedding etc.) hold ndarrays
    once populated, so this path is hit on every run after the embedder
    has filled those columns in -- must check ndarray BEFORE calling isna.
    """
    rows = []
    for _, row in df.iterrows():
        r = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, np.ndarray):
                # embedding vector -> JSON list, treat empty array as None
                r[col] = val.tolist() if val.size > 0 else None
            elif isinstance(val, (list, bool)):
                r[col] = val
            elif val is None or pd.isna(val):
                r[col] = None
            else:
                r[col] = val
        rows.append(r)
    return rows


def _try_install_pyarrow() -> bool:
    """Attempt to pip-install pyarrow silently. Returns True if successful."""
    import subprocess, importlib
    try:
        result = subprocess.run(
            ["pip", "install", "pyarrow", "-q"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            importlib.invalidate_caches()
            return True
    except Exception:
        pass
    return False


def _save_csv_fallback(df: pd.DataFrame, out: Path) -> None:
    """Save as CSV when pyarrow is unavailable. List columns pipe-delimited."""
    csv_path = out / "vector_pool.csv"
    csv_df = df.copy()
    list_cols = ["topic_tags", "algorithm_tags", "data_structure_tags",
                 "patterns", "techniques", "skill_tags",
                 "similar_problem_ids", "companies"]
    embed_cols = ["question_embedding", "solution_embedding", "rgcn_embedding",
                  "question_solution_embedding", "full_embedding"]
    for col in list_cols:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(
                lambda x: "|".join(x) if isinstance(x, list) else ""
            )
    for col in embed_cols:
        if col in csv_df.columns:
            csv_df[col] = None
    csv_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[[OK]] CSV (fallback) -> {csv_path}  ({len(csv_df)} rows)")


def save_outputs(df: pd.DataFrame, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parquet -- primary format for ML pipelines.
    # Falls back to CSV if pyarrow/fastparquet are not installed.
    parquet_path = out / "vector_pool.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[[OK]] Parquet        -> {parquet_path}  ({len(df)} rows)")
    except ImportError:
        print("[!]  pyarrow not installed. Attempting auto-install...")
        if _try_install_pyarrow():
            try:
                df.to_parquet(parquet_path, index=False)
                print(f"[[OK]] Parquet        -> {parquet_path}  ({len(df)} rows)")
            except Exception as e:
                print(f"[!]  Parquet save failed after install ({e}). Saving as CSV.")
                _save_csv_fallback(df, out)
        else:
            print("[!]  Auto-install failed. Saving as CSV instead.")
            print("     To enable Parquet later: pip install pyarrow")
            _save_csv_fallback(df, out)

    # JSON -- human-readable inspection (optional artifact)
    # Wrapped in try/except: the clean parquet is already written above,
    # so a filesystem or conversion error here should warn and continue,
    # not abort the pipeline and make the run look like a failure.
    json_path = out / "vector_pool.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(df), f, indent=2, ensure_ascii=False)
        print(f"[[OK]] JSON           -> {json_path}")
    except Exception as e:
        print(f"[!]  JSON raw dump failed ({e.__class__.__name__}: {e}) -- skipping."
              f"\n     The parquet output is complete; JSON is optional for inspection only.")

    # Schema report -- column audit
    report_path = out / "schema_report.txt"
    with open(report_path, "w") as f:
        f.write("=== DSA Engine Vector Pool -- Schema Report ===\n\n")
        f.write(f"Total problems : {len(df)}\n")
        f.write(f"Total columns  : {len(df.columns)}\n\n")
        f.write(f"{'Column':<38} {'dtype':<14} {'nulls':>6}  {'coverage':>8}\n")
        f.write("-" * 72 + "\n")
        for col in df.columns:
            dtype      = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            coverage   = f"{100*(len(df)-null_count)/max(len(df),1):.1f}%"
            f.write(f"  {col:<36} {dtype:<14} {null_count:>6}  {coverage:>8}\n")
    print(f"[[OK]] Schema report  -> {report_path}")


# ---------------------------------------------------------------------------
# Console preview
# ---------------------------------------------------------------------------

def preview(df: pd.DataFrame, n: int = 3) -> None:
    print(f"\n{'='*64}")
    print(f"  VECTOR POOL PREVIEW  ({min(n, len(df))} of {len(df)} records)")
    print(f"{'='*64}")
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        print(f"\n-- Record {i+1}: {row['title']}  [{row['problem_id']}]")
        print(f"  title_slug       : {row['title_slug']}")
        print(f"  difficulty_score : {row['difficulty_score']}")
        print(f"  frequency        : {row['frequency']}")
        print(f"  rating           : {row['rating']}")
        print(f"  likes/dislikes   : {row['likes']} / {row['dislikes']}")
        print(f"  asked_by_faang   : {row['asked_by_faang']}")
        print(f"  topic_tags       : {row['topic_tags']}")
        print(f"  algorithm_tags   : {row['algorithm_tags']}")
        print(f"  data_structure   : {row['data_structure_tags']}")
        print(f"  patterns         : {row['patterns']}")
        print(f"  techniques       : {row['techniques']}")
        print(f"  skill_tags (n)   : {len(row['skill_tags'])}")
        print(f"  companies (n)    : {len(row['companies'])}")
        print(f"  similar_ids (n)  : {len(row['similar_problem_ids'])}")
        print(f"  solution_sig     : {row['solution_signature']}")
        sol = row["canonical_solution"]
        if sol:
            print(f"  canonical_sol    : {(sol[:100]+'...') if len(sol)>100 else sol}")
        else:
            print(f"  canonical_sol    : None  <- WILL BE REJECTED BY G4")
        desc = row["description"]
        print(f"  description      : {(desc[:100]+'...') if len(desc)>100 else desc}")
        for ec in ["question_embedding","solution_embedding","rgcn_embedding",
                   "question_solution_embedding","full_embedding"]:
            print(f"  {ec:<36}: {row[ec]}")
    print()


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input",   "-i", default=None)
    p.add_argument("--output",  "-o", default="./data/vector_pool")
    p.add_argument("--preview", "-p", type=int, default=3)
    args = p.parse_args()

    sample = Path(__file__).parent / "_sample_manifest.json"
    path = args.input or str(sample)
    records = load_manifest(path)
    print(f"[[OK]] {len(records)} records loaded")
    df = build_vector_pool(records)
    print(f"[[OK]] {df.shape[0]} rows x {df.shape[1]} columns")
    if args.preview > 0:
        preview(df, args.preview)
    save_outputs(df, args.output)
    
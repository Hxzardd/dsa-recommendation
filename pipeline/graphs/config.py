"""
DSA Engine -- RGCN Config
=========================
Single source of truth for paths and hyperparameters of the RGCN stage.

All paths are RELATIVE to PROJECT_ROOT (the directory you run the pipeline
from -- not hardcoded to any machine). Every value is overridable via an
environment variable, so dev/prod or a teammate's box need zero code edits.

    PROJECT_ROOT  -> defaults to the current working directory
    KNODE_ENV     -> "dev" (default) or "prod"

Resolution order for the embedded parquet, in order of preference:
    1. $RGCN_INPUT_PARQUET                (explicit override)
    2. $PROJECT_ROOT/data/vector_pool/vector_pool_embedded.parquet
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Qdrant / Neo4j credentials -- imported from db_env.py, the ONE place in
# the whole repo that reads .env. Never define QDRANT_URL/NEO4J_PASSWORD/etc
# locally in this file -- that would create a second source of truth that
# can silently drift from db_env.py's values.
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start

_REPO_ROOT_FOR_DB_ENV = _find_repo_root(Path(__file__).resolve().parent)
if str(_REPO_ROOT_FOR_DB_ENV) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_DB_ENV))

from db_env import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_CLUSTER_ID, QDRANT_VERSION,
    QDRANT_CLOUD_PROVIDER, QDRANT_CLOUD_REGION,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    NEO4J_INSTANCEID, NEO4J_INSTANCENAME,
)
# Backward-compat alias: existing code in this repo (sources.py etc) reads
# C.NEO4J_USER -- db_env.py's canonical name is NEO4J_USERNAME (matching
# Neo4j Aura's own credential export naming). Keep both working.
NEO4J_USER = NEO4J_USERNAME


# ---------------------------------------------------------------------------
# Roots -- everything hangs off here, nothing is machine-specific
# ---------------------------------------------------------------------------

ENV = os.getenv("KNODE_ENV", "dev")

# Default to the current working dir. Run the pipeline from your project root
# and all the relative paths below "just work" on any OS.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()


def _p(env_var: str, *default_parts: str) -> Path:
    """Env-overridable path, otherwise PROJECT_ROOT / default_parts."""
    override = os.getenv(env_var)
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT.joinpath(*default_parts)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Where problem vectors + tags come from:
#   "qdrant" (default) -> read them back from the Qdrant collection the embedder
#                         uploaded to. DB-fed; no parquet needed.
#   "parquet"          -> read vector_pool_embedded.parquet (file fallback).
FEATURE_SOURCE = os.getenv("RGCN_FEATURE_SOURCE", "qdrant")

# Qdrant collection the embedder wrote the QS vectors + tag payload into.
QDRANT_SOURCE_COLLECTION = os.getenv("QDRANT_SOURCE_COLLECTION", "dsa_problems")

INPUT_PARQUET = _p("RGCN_INPUT_PARQUET", "data", "vector_pool", "vector_pool_embedded.parquet")

# Which problem-feature column drives the RGCN's problem node features.
#   question_solution_embedding (1792)  <- default, the fused vector
#   question_embedding          (1024)
#   solution_embedding          (768)
# (For FEATURE_SOURCE=qdrant this is the vector stored in the collection.)
PROBLEM_FEATURE_COL = os.getenv("RGCN_PROBLEM_FEATURE_COL", "question_solution_embedding")

# Concept node features:
#   "centroid" (default) -> mean of member problems' feature vectors. No model
#                           download, fully reproducible, lives in the same
#                           space as problems.
#   "text"               -> SentenceTransformer text embedding of the concept
#                           name (needs internet on first run).
CONCEPT_FEATURE_MODE  = os.getenv("RGCN_CONCEPT_FEATURE_MODE", "centroid")
CONCEPT_TEXT_MODEL    = os.getenv("RGCN_CONCEPT_TEXT_MODEL", "BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# Graph DB (Neo4j) -- source of curated concept edges when GRAPH_SOURCE=neo4j
# NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE are imported from
# db_env.py at the top of this file -- not redefined here.
# ---------------------------------------------------------------------------

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

ARTIFACTS_DIR  = _p("RGCN_ARTIFACTS_DIR", "pipeline", "graphs", "rgcn_artifacts")
GRAPH_PATH     = ARTIFACTS_DIR / "graph.pt"
CONCEPT_INDEX  = ARTIFACTS_DIR / "concept_index.json"
MODEL_PATH     = ARTIFACTS_DIR / "rgcn_model.pt"
EMB_NPY        = ARTIFACTS_DIR / "rgcn_problem_embeddings.npy"
# Optional parquet artifact (the primary output is the Qdrant collections).
# Set RGCN_WRITE_PARQUET=0 to skip writing it entirely.
WRITE_PARQUET  = os.getenv("RGCN_WRITE_PARQUET", "1") not in ("0", "false", "False")
OUTPUT_PARQUET = _p("RGCN_OUTPUT_PARQUET", "data", "vector_pool", "vector_pool_rgcn.parquet")


# ---------------------------------------------------------------------------
# Qdrant -- QDRANT_URL/QDRANT_API_KEY imported from db_env.py at the top of
# this file. Only collection NAMES (not credentials) are defined here.
# ---------------------------------------------------------------------------

QDRANT_URL                  = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY              = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_RGCN      = os.getenv("QDRANT_COLLECTION_RGCN",     "problems_rgcn")
QDRANT_COLLECTION_FULL      = os.getenv("QDRANT_COLLECTION_FULL",     "problems_full")
QDRANT_COLLECTION_QUESTION  = os.getenv("QDRANT_COLLECTION_QUESTION", "problems_question")
QDRANT_COLLECTION_SOLUTION  = os.getenv("QDRANT_COLLECTION_SOLUTION", "problems_solution")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Where the curated concept edges come from (on top of the tag edges that
# always exist from the problem payload/columns):
#   "tags"       -> none; concepts/edges from tags only (self-contained default).
#   "normalized" -> Aashray's curated JSON files (problem_topic / topic_topic).
#   "neo4j"      -> the same curated edges, read live from the graph DB.
# ("parquet" is accepted as an alias for "tags" for back-compat.)
GRAPH_SOURCE = os.getenv("RGCN_GRAPH_SOURCE", "tags")
if GRAPH_SOURCE == "parquet":
    GRAPH_SOURCE = "tags"

# Folder holding the four normalized JSONs (only needed for GRAPH_SOURCE=normalized).
# These files are NOT shipped with the code -- drop them here yourself.
NORMALIZED_GRAPH_DIR = _p("RGCN_GRAPH_DIR", "question-graph", "data")
GRAPH_PROBLEM_TOPIC  = NORMALIZED_GRAPH_DIR / "problem_topic_edges.json"
GRAPH_TOPIC_TOPIC    = NORMALIZED_GRAPH_DIR / "topic_topic_edges.json"
GRAPH_TOPIC_NODES    = NORMALIZED_GRAPH_DIR / "topic_nodes.json"

# Tag columns that become "concept" nodes, with their edge weight (primary
# curated signal = 1.0, looser LLM-ish signal = 0.5).
CONCEPT_TAG_WEIGHTS = {
    "topic_tags":          1.0,
    "algorithm_tags":      1.0,
    "data_structure_tags": 1.0,
    "patterns":            0.5,
    "techniques":          0.5,
}

# Weight for a curated problem->topic edge from the normalized graph.
NORMALIZED_EDGE_WEIGHT = 1.0
# Drop topic<->topic edges with fewer than this many shared problems (the
# single-overlap pairs are mostly jaccard ~0.0004 noise).
TT_MIN_SHARED_PROBLEMS = int(os.getenv("RGCN_TT_MIN_SHARED", 2))

# Add embedding-KNN problem->problem edges on top of similar_problem_ids,
# to densify the similar_to relation. 0 disables KNN augmentation.
SIMILARITY_KNN_K        = int(os.getenv("RGCN_SIMILARITY_KNN_K", 5))
# Min co-occurrence count for a concept<->concept edge to be kept.
COOCCUR_MIN_COUNT       = int(os.getenv("RGCN_COOCCUR_MIN_COUNT", 3))


# ---------------------------------------------------------------------------
# Model + training hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM          = int(os.getenv("RGCN_HIDDEN_DIM", 256))
OUT_DIM             = int(os.getenv("RGCN_OUT_DIM", 128))
DROPOUT             = float(os.getenv("RGCN_DROPOUT", 0.2))

EPOCHS              = int(os.getenv("RGCN_EPOCHS", 400))
LR                  = float(os.getenv("RGCN_LR", 1e-3))
WEIGHT_DECAY        = float(os.getenv("RGCN_WEIGHT_DECAY", 1e-5))
EARLY_STOP_PATIENCE = int(os.getenv("RGCN_EARLY_STOP_PATIENCE", 80))

VAL_FRACTION        = float(os.getenv("RGCN_VAL_FRACTION", 0.1))
NEG_PER_POS         = int(os.getenv("RGCN_NEG_PER_POS", 2))
SIMILAR_LOSS_WEIGHT = float(os.getenv("RGCN_SIMILAR_LOSS_WEIGHT", 3.0))

SEED                = int(os.getenv("RGCN_SEED", 42))
DEVICE              = os.getenv("RGCN_DEVICE", "auto")  # "auto" | "cpu" | "cuda"


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def summary() -> str:
    return (
        f"PROJECT_ROOT        : {PROJECT_ROOT}\n"
        f"FEATURE_SOURCE      : {FEATURE_SOURCE}"
        + (f"  (collection={QDRANT_SOURCE_COLLECTION})\n" if FEATURE_SOURCE == "qdrant"
           else f"  ({INPUT_PARQUET})\n")
        + f"PROBLEM_FEATURE_COL : {PROBLEM_FEATURE_COL}\n"
        f"CONCEPT_FEATURE_MODE: {CONCEPT_FEATURE_MODE}\n"
        f"GRAPH_SOURCE        : {GRAPH_SOURCE}\n"
        f"ARTIFACTS_DIR       : {ARTIFACTS_DIR}\n"
        f"WRITE_PARQUET       : {WRITE_PARQUET}\n"
        f"QDRANT_URL          : {QDRANT_URL}\n"
        f"HIDDEN/OUT/EPOCHS   : {HIDDEN_DIM}/{OUT_DIM}/{EPOCHS}\n"
        f"DEVICE              : {resolve_device()}\n"
    )
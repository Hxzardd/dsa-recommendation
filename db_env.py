"""
db_env.py

Single source of truth for every Qdrant and Neo4j credential used across
the whole repo. Every file that touches either database gets its
credentials FROM HERE -- directly (imports this module) or indirectly
(imports pipeline/graphs/config.py, which itself imports this module).
No file should hardcode a URL, password, or fall back to a bare
os.environ.get() of its own -- this is the one place .env gets parsed.

PostgreSQL is intentionally NOT read here. The ML repo never connects to
Postgres directly for writes -- backend owns every write to every
Postgres table. The one place ML reads Postgres (UserGraphService's
SELECT-only queries) takes a `db` session object passed in by the caller,
not a credential this module would own.

Where the values actually come from:

    Qdrant Cloud cluster dashboard:
        QDRANT_URL             cluster endpoint, e.g. https://xxxx.aws.cloud.qdrant.io:6333
        QDRANT_API_KEY         cluster API key
        QDRANT_CLUSTER_ID      the cluster's ID (dashboard URL / cluster settings)
        QDRANT_VERSION         Qdrant server version the cluster runs
        QDRANT_CLOUD_PROVIDER  e.g. AWS / GCP / Azure
        QDRANT_CLOUD_REGION    e.g. us-east-1

    Neo4j Aura downloaded credentials file:
        NEO4J_URI              e.g. neo4j+s://xxxxxxxx.databases.neo4j.io
        NEO4J_USERNAME         usually "neo4j"
        NEO4J_PASSWORD
        NEO4J_DATABASE         usually "neo4j" (Aura's default database name)
                                -- the ONLINE per-user graph (:User/:Concept/
                                :CC_EDGE, written by Neo4jGraphStore).
        NEO4J_OFFLINE_DATABASE  optional -- a SEPARATE Neo4j database name
                                for the offline concept graph the RGCN
                                pipeline generates (topic-topic co-occurrence
                                /prerequisite structure), kept apart from
                                per-user online state so a full offline
                                pipeline re-run never touches live user
                                graphs and vice versa. Defaults to
                                NEO4J_DATABASE if unset (same database, but
                                pipeline/graphs/neo4j_offline_writer.py still
                                uses distinct :OfflineTopic labels /
                                CO_OCCURS_OFFLINE-PREREQ_OFFLINE relationship
                                types so the two stay visually and
                                query-wise separated even then -- Neo4j
                                Community Edition only supports one database
                                per instance, so the label/relationship-type
                                separation is what actually matters there).
        AURA_INSTANCEID        (aliased as NEO4J_INSTANCEID too, either works)
        AURA_INSTANCENAME      (aliased as NEO4J_INSTANCENAME too, either works)
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk upward from `start` looking for pyproject.toml -- robust to
    however deep the calling file sits in the repo tree."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start   # fallback: caller's own directory


def _load_dotenv(path: Path) -> None:
    """
    Tries python-dotenv first; falls back to a tiny manual KEY=VALUE
    parser so this module doesn't hard-require an extra dependency.
    Never overwrites a variable already set in the real environment
    (matches python-dotenv's default behaviour).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
_load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL            = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY        = os.environ.get("QDRANT_API_KEY")
QDRANT_CLUSTER_ID     = os.environ.get("QDRANT_CLUSTER_ID")
QDRANT_VERSION        = os.environ.get("QDRANT_VERSION")
QDRANT_CLOUD_PROVIDER = os.environ.get("QDRANT_CLOUD_PROVIDER")
QDRANT_CLOUD_REGION   = os.environ.get("QDRANT_CLOUD_REGION")

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
NEO4J_URI          = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME     = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD     = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE     = os.environ.get("NEO4J_DATABASE", "neo4j")

# Offline concept graph (RGCN-generated topic-topic structure) -- kept
# separate from NEO4J_DATABASE's online per-user state. Defaults to the
# same database as NEO4J_DATABASE if unset; pipeline/graphs/
# neo4j_offline_writer.py additionally uses distinct :OfflineTopic labels
# and CO_OCCURS_OFFLINE/PREREQ_OFFLINE relationship types regardless, so
# the separation holds even on Neo4j Community Edition (one database per
# instance only) -- see this module's docstring for the full rationale.
NEO4J_OFFLINE_DATABASE = os.environ.get("NEO4J_OFFLINE_DATABASE", NEO4J_DATABASE)
NEO4J_INSTANCEID   = os.environ.get("NEO4J_INSTANCEID")   or os.environ.get("AURA_INSTANCEID")
NEO4J_INSTANCENAME = os.environ.get("NEO4J_INSTANCENAME") or os.environ.get("AURA_INSTANCENAME")


# ---------------------------------------------------------------------------
# Convenience factories -- every script should use these instead of
# constructing a QdrantClient / Neo4j driver by hand, so a credential
# change here is the only place that needs updating.
# ---------------------------------------------------------------------------

def qdrant_client(timeout: int = 10):
    """Returns a QdrantClient built from the credentials above."""
    import warnings
    from qdrant_client import QdrantClient

    # Suppress the "Failed to obtain server version" UserWarning that
    # Qdrant Cloud triggers on Windows -- it's cosmetic (the client works
    # fine) but noisy. We suppress it rather than using check_compatibility=False
    # because that parameter was added in a later qdrant-client version and
    # crashes on older installs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=timeout,
        )


def neo4j_driver():
    """
    Returns a connected Neo4j driver, or None if NEO4J_PASSWORD isn't set
    or the connection fails. Callers should treat None as "Neo4j disabled
    for this run" -- matches Neo4jGraphStore(driver=None)'s existing
    no-op behaviour, never a crash.

    FIX (Greptile P1 "Missing Driver Silently Disables Neo4j"): the
    `neo4j` package was missing from pyproject.toml/uv.lock, so a normal
    locked install raised ModuleNotFoundError here -- caught by the same
    broad `except Exception` as a real connection failure, so a missing
    dependency was silently indistinguishable from Neo4j just being
    unreachable. neo4j>=6.0.0 is now in pyproject.toml (run `uv lock` to
    regenerate uv.lock if you haven't). This also now logs a distinct,
    actionable message for the missing-package case specifically, so
    this doesn't go silent again if the lockfile ever drifts.
    """
    if not NEO4J_PASSWORD:
        return None
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError:
        import logging
        logging.getLogger(__name__).error(
            "neo4j package not installed -- Neo4j durable storage disabled. "
            "Run: uv add neo4j  (or confirm neo4j>=6.0.0 is in pyproject.toml "
            "and re-run: uv lock && uv sync)"
        )
        return None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as exc:
        # FIX: this used to swallow the exception entirely -- every failure
        # mode (wrong password, DNS failure, firewall block, paused Aura
        # instance, expired credentials) looked identical to every caller:
        # "Neo4j unreachable", no way to tell which. Logs the real
        # exception class + message (Neo4j driver errors don't echo the
        # password back, so this is safe to log) so a caller actually
        # knows what to fix instead of guessing.
        import logging
        logging.getLogger(__name__).error(
            "Neo4j connection failed (%s: %s) -- check NEO4J_URI/NEO4J_USERNAME/"
            "NEO4J_PASSWORD, that the Aura instance isn't paused, and that this "
            "machine can reach %s.",
            exc.__class__.__name__, exc, NEO4J_URI,
        )
        return None


def describe() -> dict:
    """Non-secret summary for logging/debugging -- never includes the API key or password."""
    return {
        "qdrant_url":            QDRANT_URL,
        "qdrant_cluster_id":     QDRANT_CLUSTER_ID,
        "qdrant_version":        QDRANT_VERSION,
        "qdrant_cloud_provider": QDRANT_CLOUD_PROVIDER,
        "qdrant_cloud_region":   QDRANT_CLOUD_REGION,
        "qdrant_api_key_set":    bool(QDRANT_API_KEY),
        "neo4j_uri":             NEO4J_URI,
        "neo4j_username":        NEO4J_USERNAME,
        "neo4j_database":        NEO4J_DATABASE,
        "neo4j_instanceid":      NEO4J_INSTANCEID,
        "neo4j_instancename":    NEO4J_INSTANCENAME,
        "neo4j_password_set":    bool(NEO4J_PASSWORD),
    }
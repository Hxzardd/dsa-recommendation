"""
Maps this ML service's topic slugs onto the backend's `topic` table `slug`
column.

As of the backend's taxonomy-reconciliation PR, the backend's `topic`
table holds exactly the 70 canonical tags from
data/topic_tags_taxonomy_v2.json, hyphenated (e.g. "dynamic_programming"
-> "dynamic-programming") -- verified directly against a live query of the
`topic` table: all 70 canonical tags, hyphenated, match all 70 live
`topic.slug` values exactly, no gaps either direction. So the mapping is
now a straight, total, bijective transform (canonical_tag <-> topic.slug
via underscore<->hyphen) -- no hand-curated per-topic table needed
anymore (the old version of this file hard-coded a 35-entry table against
the backend's PRE-reconciliation 44-topic schema; that schema no longer
exists live and every lookup against it silently returned None).

The ML pipeline's OWN topic tags (Qdrant topic_tags, BKT/HLR mastery
keys) are being migrated to use these same 70 canonical tags directly. For
backward compatibility with any pre-migration data still using the old,
finer-grained 72-tag taxonomy (data/problem_topic_edges_normalized.json's
previous tag set), `legacy_and_ai_taxonomy_map` in the taxonomy file
collapses each old tag to its (possibly multiple) canonical equivalent(s)
-- this module uses the first/primary one for FK-write purposes, matching
the backend's own example (two_pass_scan -> greedy).
"""

from __future__ import annotations

import json
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_TAXONOMY_PATH = _BASE_DIR / "data" / "topic_tags_taxonomy_v2.json"

with open(_TAXONOMY_PATH, encoding="utf-8-sig") as f:
    _TAXONOMY = json.load(f)

CANONICAL_TAGS = frozenset(_TAXONOMY["canonical_tags"])

# legacy_and_ai_taxonomy_map ships without an entry for "two_pass_scan" --
# confirmed the one gap in an otherwise-complete map covering all 72 old
# ML tags. Backend's own review named the exact collapse target.
_LEGACY_TAG_MAP: dict = dict(_TAXONOMY["legacy_and_ai_taxonomy_map"])
_LEGACY_TAG_MAP.setdefault("two_pass_scan", ["greedy"])


def canonical_for_ml_slug(ml_slug: str) -> str | None:
    """
    Any ML-pipeline topic slug -> its canonical tag. Already-canonical
    slugs pass through unchanged; legacy (pre-migration) slugs collapse to
    their primary canonical equivalent via the taxonomy's own map. None if
    the slug is neither -- callers must skip the write, not guess.
    """
    if ml_slug in CANONICAL_TAGS:
        return ml_slug
    mapped = _LEGACY_TAG_MAP.get(ml_slug)
    return mapped[0] if mapped else None


def topic_slug_for_ml_slug(ml_slug: str) -> str | None:
    """ML-pipeline slug (legacy or canonical, e.g. "two_pass_scan" or
    "array") -> backend topic table slug (e.g. "greedy" or "array" with
    underscores replaced by hyphens, e.g. "dynamic-programming")."""
    canonical = canonical_for_ml_slug(ml_slug)
    return canonical.replace("_", "-") if canonical else None


def ml_slug_for_topic_slug(topic_slug: str) -> str | None:
    """Reverse of topic_slug_for_ml_slug -- backend topic.slug -> the
    canonical ML tag (this is total/bijective post-reconciliation, so
    every live topic.slug has a valid ML-slug counterpart)."""
    canonical = topic_slug.replace("-", "_")
    return canonical if canonical in CANONICAL_TAGS else None


_LEETCODE_TAG_MAP: dict = _TAXONOMY["official_leetcode_tag_map"]


def canonical_tags_for_leetcode_slug(lc_slug: str) -> list[str]:
    """LeetCode's own topicTags slug (e.g. "hash-table", as returned by
    their GraphQL API) -> our canonical ML tag(s) (e.g. ["hash_map"]).
    Used by seeding_controller.py's LeetCode history import -- LC's tags
    are a different taxonomy from ours and need this translation before
    they're usable as BKT/HLR topic keys. Empty list if LC's slug has no
    known mapping (unrecognised/new LC tag)."""
    return list(_LEETCODE_TAG_MAP.get(lc_slug, []))

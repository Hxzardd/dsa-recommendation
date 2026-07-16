"""
Maps this ML service's topic taxonomy (the underscore/singular slugs used by
BKT/HLR/Qdrant's topic_tags -- see data/problem_topic_edges_normalized.json
and data/source-target.txt, which are identical) onto the backend's `topic`
table `slug` column (hyphenated, and sometimes pluralised/differently named --
e.g. "arrays" not "array", "dynamic-programming" not "dp").

Every entry below was derived by diffing the ML taxonomy's real tag set
against a live query of the `topic` table's actual `slug` values, keeping
only unambiguous normalisation matches (hyphen<->underscore, singular<->
plural, and a handful of confirmed synonyms like dynamic-programming/dp,
graphs/graph). Nothing here is guessed -- see ML_SLUG_TO_TOPIC_SLUG's
construction history for the verification query.

Nine backend topics have NO ML-taxonomy counterpart at all (the ML pipeline
never emits a topic for these): binary-indexed-tree, binary-lifting,
binary-search, binary-search-tree, geometry, priority-queue, recursion,
segment-tree, topological-sort. mastery/HLR data can never be written for
these topics via the ML pipeline as it exists today -- that's a real gap
between the two taxonomies, not something this file can safely paper over
by guessing a match.

This service never migrates the `topic` table (see dev_schema.sql's header
comment -- it's backend-owned); this mapping only translates at read/write
time via the table's existing `slug` column.
"""

from __future__ import annotations

ML_SLUG_TO_TOPIC_SLUG = {
    "array":               "arrays",
    "backtracking":        "backtracking",
    "bfs":                 "bfs",
    "bitmask_dp":          "bitmask-dp",
    "bitwise":             "bitwise",
    "design":              "design",
    "dfs":                 "dfs",
    "digit_dp":            "digit-dp",
    "dijkstra":            "dijkstra",
    "dp":                  "dynamic-programming",
    "fast_slow_pointers":  "fast-slow-pointers",
    "floyd_warshall":      "floyd-warshall",
    "graph":               "graphs",
    "greedy":              "greedy",
    "hash_map":            "hash-map",
    "heap":                "heap",
    "kmp":                 "kmp",
    "linked_list":         "linked-lists",
    "math":                "math",
    "merge_intervals":     "merge-intervals",
    "monotonic_stack":     "monotonic-stack",
    "number_theory":       "number-theory",
    "prefix_sum":          "prefix-sum",
    "queue":               "queue",
    "shortest_path":       "shortest-path",
    "simulation":          "simulation",
    "sliding_window":      "sliding-window",
    "stack":               "stacks",
    "string_matching":     "string-matching",
    "string":              "strings",
    "tree_dp":             "tree-dp",
    "tree":                "trees",
    "trie":                "trie",
    "two_pointers":        "two-pointers",
    "union_find":          "union-find",
}

# Backend topic.slug values with no ML-taxonomy counterpart -- documented so
# callers can tell "no mapping exists" apart from "mapping lookup bug".
UNMAPPED_TOPIC_SLUGS = frozenset({
    "binary-indexed-tree", "binary-lifting", "binary-search",
    "binary-search-tree", "geometry", "priority-queue", "recursion",
    "segment-tree", "topological-sort",
})


TOPIC_SLUG_TO_ML_SLUG = {v: k for k, v in ML_SLUG_TO_TOPIC_SLUG.items()}


def topic_slug_for_ml_slug(ml_slug: str) -> str | None:
    """Translate an ML-pipeline topic slug (e.g. "array") to the backend
    topic table's slug (e.g. "arrays"). Returns None if the ML pipeline
    emits a slug with no confirmed backend counterpart (e.g. fine-grained
    pattern tags like "kadane", "two_heaps" that don't correspond to any
    of the backend's 44 coarser topic rows) -- callers must handle None by
    skipping the write, not guessing a topic_id.
    """
    return ML_SLUG_TO_TOPIC_SLUG.get(ml_slug)


def ml_slug_for_topic_slug(topic_slug: str) -> str | None:
    """Reverse of topic_slug_for_ml_slug -- translate a backend topic
    table slug back to the ML-pipeline slug UserGraph/pools expect. None
    for the 9 UNMAPPED_TOPIC_SLUGS."""
    return TOPIC_SLUG_TO_ML_SLUG.get(topic_slug)

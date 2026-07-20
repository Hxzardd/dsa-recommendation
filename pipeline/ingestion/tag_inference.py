"""
DSA Engine — Tag Inference
===========================
Derives all tag fields from the raw problem record using
plain literal substring matching. All keywords are lowercased
and matched against a lowercased text blob — no regex, no escaping.
"""

from __future__ import annotations
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Search text builder
# ---------------------------------------------------------------------------

def _search_text(record: Dict[str, Any]) -> str:
    """Single lowercase blob of all useful text in a record."""
    parts = [
        record.get("title", ""),
        record.get("explanation_text", ""),
        record.get("full_markdown_solution", ""),
        record.get("python_solution", ""),
        record.get("python_solution_eri", ""),
    ]
    for cs in (record.get("community_solutions") or []):
        parts.append(cs.get("post_title", ""))
        parts.append(cs.get("code", ""))
    for sq in (record.get("similar_questions") or []):
        if isinstance(sq, str):
            parts.append(sq)
    return " ".join(p for p in parts if p).lower()


# ---------------------------------------------------------------------------
# Keyword tables  (keyword, [tags])
# All keywords already lowercase — matched against lowercased text.
# ---------------------------------------------------------------------------

_TOPIC_KW: List[tuple] = [
    ("array",               ["arrays"]),
    ("nums[",               ["arrays"]),
    ("height[",             ["arrays"]),
    ("prices[",             ["arrays"]),
    ("string",              ["strings"]),
    ("str(",                ["strings"]),
    ("char",                ["strings"]),
    ("substring",           ["strings"]),
    ("palindrome",          ["strings", "palindromes"]),
    ("linked list",         ["linked_lists"]),
    ("listnode",            ["linked_lists"]),
    ("binary tree",         ["trees"]),
    ("treenode",            ["trees"]),
    ("inorder",             ["trees"]),
    ("preorder",            ["trees"]),
    ("postorder",           ["trees"]),
    ("graph",               ["graphs"]),
    ("adjacency",           ["graphs"]),
    ("dynamic programming", ["dynamic_programming"]),
    ("dp[",                 ["dynamic_programming"]),
    ("dp =",                ["dynamic_programming"]),
    (" dp ",                ["dynamic_programming"]),
    ("memoiz",              ["dynamic_programming"]),
    ("binary search",       ["binary_search"]),
    ("backtrack",           ["backtracking"]),
    ("sort",                ["sorting"]),
    ("interval",            ["intervals"]),
    ("stack",               ["stacks"]),
    ("queue",               ["queues"]),
    ("heap",                ["heaps"]),
    ("heapq",               ["heaps"]),
    ("priority queue",      ["heaps"]),
    ("trie",                ["tries"]),
    ("hash",                ["hash_tables"]),
    ("matrix",              ["matrices"]),
    ("two pointer",         ["two_pointers"]),
    ("two-pointer",         ["two_pointers"]),
    ("sliding window",      ["sliding_window"]),
    ("bit manipulation",    ["bit_manipulation"]),
    ("bitwise",             ["bit_manipulation"]),
    ("math",                ["math"]),
    ("recursion",           ["recursion"]),
    ("recursive",           ["recursion"]),
    ("greedy",              ["greedy"]),
]

_ALGORITHM_KW: List[tuple] = [
    ("binary search",       ["binary_search"]),
    ("two pointer",         ["two_pointers"]),
    ("two-pointer",         ["two_pointers"]),
    ("sliding window",      ["sliding_window"]),
    ("dynamic programming", ["dynamic_programming"]),
    ("dp[",                 ["dynamic_programming"]),
    (" dp ",                ["dynamic_programming"]),
    ("dp =",                ["dynamic_programming"]),
    ("backtrack",           ["backtracking"]),
    ("depth-first",         ["dfs"]),
    ("dfs(",                ["dfs"]),
    ("inorder",             ["dfs"]),
    ("preorder",            ["dfs"]),
    ("postorder",           ["dfs"]),
    ("breadth-first",       ["bfs"]),
    ("bfs",                 ["bfs"]),
    ("level order",         ["bfs"]),
    ("divide and conquer",  ["divide_and_conquer"]),
    ("greedy",              ["greedy"]),
    ("topological",         ["topological_sort"]),
    ("union find",          ["union_find"]),
    ("disjoint set",        ["union_find"]),
    ("bit manipulation",    ["bit_manipulation"]),
    ("bitwise",             ["bit_manipulation"]),
    ("memoiz",              ["memoization"]),
    ("lru_cache",           ["memoization"]),
    ("recursion",           ["recursion"]),
    ("recursive",           ["recursion"]),
    ("sorted(",             ["sorting"]),
    (".sort(",              ["sorting"]),
    ("merge sort",          ["sorting"]),
    ("prefix sum",          ["prefix_sums"]),
    ("prefix",              ["prefix_sums"]),
    ("kadane",              ["kadane"]),
    ("hash map",            ["hashing"]),
    ("hashmap",             ["hashing"]),
    ("hash table",          ["hashing"]),
    ("counter(",            ["hashing"]),
    ("two sum",             ["hashing"]),
    ("enumerate(",          ["iteration"]),
    ("for i",               ["iteration"]),
    ("for num",             ["iteration"]),
    ("for val",             ["iteration"]),
    ("while",               ["iteration"]),
    ("iterate",             ["iteration"]),
    ("traverse",            ["traversal"]),
    ("pointer",             ["pointer_manipulation"]),
    ("linked",              ["linked_list_traversal"]),
    (".next",               ["linked_list_traversal"]),
    ("simulation",          ["simulation"]),
    ("simulate",            ["simulation"]),
    ("sliding",             ["sliding_window"]),
    ("dijkstra",            ["shortest_path"]),
    ("bellman",             ["shortest_path"]),
    ("prim",                ["minimum_spanning_tree"]),
    ("floyd",               ["cycle_detection"]),
    ("fast and slow",       ["fast_slow_pointers"]),
    ("fast, slow",          ["fast_slow_pointers"]),
]

_DATA_STRUCTURE_KW: List[tuple] = [
    ("hash map",            ["hash_map"]),
    ("hashmap",             ["hash_map"]),
    ("unordered_map",       ["hash_map"]),
    ("hash table",          ["hash_table"]),
    ("defaultdict",         ["hash_map"]),
    ("counter(",            ["hash_map"]),
    ("dictionary",          ["hash_map"]),
    ("d = {}",              ["hash_map"]),
    ("map = {}",            ["hash_map"]),
    ("stack",               ["stack"]),
    ("deque",               ["deque"]),
    ("queue",               ["queue"]),
    ("linked list",         ["linked_list"]),
    ("listnode",            ["linked_list"]),
    ("binary tree",         ["binary_tree"]),
    ("treenode",            ["binary_tree"]),
    ("binary search tree",  ["bst"]),
    (" bst",                ["bst"]),
    ("heap",                ["heap"]),
    ("heapq",               ["heap"]),
    ("priority queue",      ["priority_queue"]),
    ("trie",                ["trie"]),
    ("segment tree",        ["segment_tree"]),
    ("fenwick",             ["fenwick_tree"]),
    ("graph",               ["graph"]),
    ("adjacency",           ["graph"]),
    ("matrix",              ["matrix"]),
    ("nums[",               ["array"]),
    ("height[",             ["array"]),
    ("prices[",             ["array"]),
    ("array",               ["array"]),
    ("set()",               ["set"]),
    ("string",              ["string"]),
    ("str(",                ["string"]),
    ("char",                ["string"]),
]

_PATTERN_KW: List[tuple] = [
    # Pointer patterns
    ("two pointer",         ["two_pointers"]),
    ("two-pointer",         ["two_pointers"]),
    ("left, right",         ["two_pointers"]),
    ("fast and slow",       ["fast_slow_pointers"]),
    ("fast, slow",          ["fast_slow_pointers"]),
    ("slow",                ["fast_slow_pointers"]),    # covers: slow/fast pointer problems
    ("fast",                ["fast_slow_pointers"]),    # 938 hits — biggest gap
    ("floyd",               ["fast_slow_pointers"]),
    ("cycle",               ["cycle_detection"]),
    # Window / search
    ("sliding window",      ["sliding_window"]),
    ("window",              ["sliding_window"]),
    ("binary search",       ["binary_search_pattern"]),
    ("prefix",              ["prefix_sums"]),
    ("prefix sum",          ["prefix_sums"]),
    ("subarray",            ["subarray_pattern"]),
    ("kadane",              ["kadane"]),
    # Hash / lookup
    ("hash map",            ["hash_map_lookup"]),
    ("complement",          ["complement_lookup"]),
    # Tree traversal
    ("inorder",             ["tree_dfs"]),
    ("preorder",            ["tree_dfs"]),
    ("postorder",           ["tree_dfs"]),
    ("level order",         ["tree_bfs"]),
    ("bfs",                 ["tree_bfs"]),
    # Stack patterns
    ("stack",               ["stack_pattern"]),
    ("parenthes",           ["matching_brackets"]),
    ("monotonic",           ["monotonic_stack"]),
    # DP patterns
    ("dp[",                 ["dp_tabulation"]),
    ("dp =",                ["dp_tabulation"]),
    ("memo",                ["memoization_pattern"]),
    ("memoiz",              ["memoization_pattern"]),
    ("lru_cache",           ["memoization_pattern"]),
    ("fibonacci",           ["fibonacci"]),
    ("knapsack",            ["knapsack"]),
    # Graph patterns
    ("topological",         ["topological_sort"]),
    ("union find",          ["union_find"]),
    ("dijkstra",            ["shortest_path"]),
    ("backtrack",           ["backtracking_pattern"]),
    # Merge / divide
    ("merge",               ["merge_pattern"]),
    ("divide and conquer",  ["divide_and_conquer"]),
    # Interval
    ("merge interval",      ["merge_intervals"]),
    ("interval",            ["intervals_pattern"]),
    # Linked list patterns
    ("dummy",               ["dummy_node_pattern"]),
    ("reverse",             ["in_place_reversal"]),
    # Trie
    ("trie",                ["trie_pattern"]),
    # Other
    ("top k",               ["top_k_elements"]),
    ("k-way merge",         ["k_way_merge"]),
    ("subset",              ["subsets"]),
    ("combination",         ["subsets"]),
    ("two heap",            ["two_heaps"]),
    ("palindrome",          ["palindrome"]),
]

_TECHNIQUE_KW: List[tuple] = [
    # Memoization
    ("memoiz",              ["memoization"]),
    ("lru_cache",           ["memoization"]),
    # DP direction
    ("bottom-up",           ["bottom_up_dp"]),
    ("bottom up",           ["bottom_up_dp"]),
    ("tabulation",          ["tabulation"]),
    ("top-down",            ["top_down_dp"]),
    ("top down",            ["top_down_dp"]),
    # Pointer techniques
    ("prev",                ["pointer_rewiring"]),      # 840 hits — was completely missing
    ("dummy",               ["dummy_node"]),
    ("sentinel",            ["sentinel_node"]),
    ("floyd",               ["floyd_cycle_detection"]),
    # Math techniques
    ("% 10",                ["digit_extraction"]),
    ("divmod",              ["divmod_technique"]),
    ("mod",                 ["modulo_technique"]),      # 589 hits — was missing
    ("carry",               ["carry_technique"]),
    # Bit tricks
    ("xor",                 ["xor_trick"]),
    ("bit",                 ["bit_manipulation_technique"]),
    # State tracking
    ("seen",                ["seen_set"]),              # 308 hits — was missing
    ("visited",             ["visited_set"]),           # 294 hits — was missing
    # In-place
    ("in-place",            ["in_place"]),
    ("in place",            ["in_place"]),
    ("swap",                ["swap_technique"]),        # 171 hits — was missing
    # Algorithmic techniques
    ("divide and conquer",  ["divide_and_conquer"]),
    ("two pass",            ["two_pass"]),
    ("pruning",             ["pruning"]),
    ("fibonacci",           ["fibonacci_technique"]),
]


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

def _match(text: str, kw_list: List[tuple]) -> List[str]:
    found: set = set()
    for kw, tags in kw_list:
        if kw in text:
            found.update(tags)
    return sorted(found)


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

def infer_topic_tags(record: Dict[str, Any]) -> List[str]:
    return _match(_search_text(record), _TOPIC_KW)

def infer_algorithm_tags(record: Dict[str, Any]) -> List[str]:
    return _match(_search_text(record), _ALGORITHM_KW)

def infer_data_structure_tags(record: Dict[str, Any]) -> List[str]:
    return _match(_search_text(record), _DATA_STRUCTURE_KW)

def infer_patterns(record: Dict[str, Any]) -> List[str]:
    return _match(_search_text(record), _PATTERN_KW)

def infer_techniques(record: Dict[str, Any]) -> List[str]:
    return _match(_search_text(record), _TECHNIQUE_KW)

def infer_skill_tags(record: Dict[str, Any]) -> List[str]:
    combined = set(
        infer_topic_tags(record)
        + infer_algorithm_tags(record)
        + infer_data_structure_tags(record)
        + infer_patterns(record)
        + infer_techniques(record)
    )
    return sorted(combined)


# ---------------------------------------------------------------------------
# Difficulty score
# ---------------------------------------------------------------------------

def infer_difficulty_score(record: Dict[str, Any]) -> float:
    rating = record.get("rating")
    freq   = record.get("frequency")
    if rating is not None:
        try:
            r = float(rating)
            # Compress rating (a like-ratio, 0-1, higher = better-liked/
            # easier) into the target [0.1, 0.9] difficulty range.
            # BUG (fixed): the previous formula was `1.0 - r*0.7 + 0.1`
            # == `1.1 - 0.7*r`, which only reaches 0.4 at r=1.0 (the
            # best-liked, easiest problems) instead of the intended 0.1 --
            # the 0.1 floor was structurally unreachable for ANY rating
            # value, so no problem in the catalog could ever be scored
            # below ~0.4 difficulty. Confirmed empirically: a live query of
            # the ingested Qdrant collection found zero problems below
            # 0.34 difficulty catalog-wide. `0.9 - r*0.8` correctly spans
            # the full target range: r=0 -> 0.9 (least-liked/hardest),
            # r=1 -> 0.1 (best-liked/easiest).
            return round(max(0.1, min(0.9, 0.9 - r * 0.8)), 3)
        except (TypeError, ValueError):
            pass
    if freq is not None:
        try:
            return round(min(float(freq) * 0.8, 0.95), 3)
        except (TypeError, ValueError):
            pass
    return 0.5


# ---------------------------------------------------------------------------
# Solution utilities
# ---------------------------------------------------------------------------

def infer_solution_signature(record: Dict[str, Any]) -> str:
    sol = pick_canonical_solution(record)
    if not sol:
        return ""
    for line in sol.splitlines():
        s = line.strip()
        if s.startswith("def ") or s.startswith("class "):
            return s
    return ""


def pick_canonical_solution(record: Dict[str, Any]):
    """
    Priority chain:
      1. python_solution       (community-voted, highest quality)
      2. python_solution_eri   (ERI reference)
      3. highest-upvoted community_solutions[].code
      4. None                  (SQL/Shell — no Python solution)
    """
    ps = (record.get("python_solution") or "").strip()
    if ps:
        return ps

    eri = (record.get("python_solution_eri") or "").strip()
    if eri:
        return eri

    community = record.get("community_solutions") or []
    if community:
        best = max(community, key=lambda x: int(x.get("upvotes", 0)))
        code = (best.get("code") or "").strip()
        if code:
            return code

    return None


def extract_similar_problem_ids(record: Dict[str, Any]) -> List[str]:
    """
    Extract problem slugs from similar_questions.

    Two formats exist in the dataset:
      A. ['[3Sum', '/problems/3sum/', 'Medium]', ...]  → extract /problems/ entries
      B. ['nan']                                        → no real data, return []
    """
    raw = record.get("similar_questions") or []
    ids = []
    for item in raw:
        if not isinstance(item, str):
            continue
        item = item.strip()
        # Skip nan placeholders
        if item.lower() == "nan" or item == "['nan']":
            continue
        # Extract from /problems/<slug>/ entries
        if item.startswith("/problems/"):
            slug = item.strip("/").split("/")[-1]
            if slug:
                ids.append(slug)
    return ids

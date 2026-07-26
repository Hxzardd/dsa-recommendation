# Labeling Strategy for LambdaRank Training

## Available data (no invented signals)

Every label derives from two existing tables, both already read elsewhere
in this codebase:

- `recommendation_log` (`database/postgres/db.py::save_recommendation_log`) —
  one row per recommended problem, with `was_attempted` flipped to `TRUE` by
  `mark_recommendation_attempted` (called from `submission_controller.py::handle_update`
  on **any** completed submission, pass or fail).
- `submission` (already read by
  `pipeline/recommender/services/user_graph_service.py::_load_submissions`) —
  gives `verdict` and `submitted_at` per attempt, which `recommendation_log`
  alone cannot: `was_attempted` only means "a submission happened," not
  "it was solved."

### A field that exists but carries no signal

`recommendation_log.was_skipped` and `skip_count` are written as `FALSE`/`0`
at insert time and **read** by `user_graph_service.py::_load_recommendation_log`,
but grep confirms nothing anywhere in this codebase ever updates them after
insertion. They are always `FALSE`/`0` today. **Do not use them as if they
carry real signal** — `label_generator.py` derives "ignored" purely from
`was_attempted=False` after the configured window elapses, not from these
fields.

**Future logging improvement**: if genuine skip-detection is ever
implemented (e.g., the frontend logging "user viewed this recommendation
and dismissed it" or "scrolled past without opening"), that would be a much
stronger negative signal than time-based "never attempted," and should
replace the window-based ignore heuristic once available.

## Strategies considered

| Strategy | Labels | Pros | Cons |
|---|---|---|---|
| **Binary (solved)** | 0/1 | Simple, unambiguous | Collapses "attempted and failed" with "completely ignored" — throws away a real, already-available distinction. Weaker ranking signal for LambdaRank, which benefits from graded relevance to learn finer orderings (NDCG rewards correct relative ordering across relevance levels, not just a coin-flip). |
| **Ternary engagement (chosen default)** | 0/1/2 | Uses exactly the distinctions the existing data actually supports (no submission at all vs. submitted-but-failed vs. solved) without inventing anything finer. | Doesn't distinguish "solved on the 1st try" from "solved on the 10th." |
| **Graded by solve speed/attempt count** | 0–4 | Richest possible signal on paper. | **Rejected.** This recommender's own ranking philosophy already treats "solved instantly" as *not* more relevant than a productive-struggle solve — `HeuristicRanker`'s `proximity` term peaks at `predicted_success = 0.68`, not `1.0`; the ZPD band (`0.55`–`0.80`) explicitly excludes near-certain successes as "too easy, no growth." Rewarding raw speed/low-attempt-count in the *label* would train a future LambdaRank model in the opposite direction from what the current system already believes is good recommending, based on an invented "ideal attempt curve" the data doesn't itself justify. |

## Chosen default: `TernaryEngagementLabelStrategy`

```
0 = never attempted within the window (ignored)
1 = attempted, but not solved within the window
2 = solved within the window
```

This is the richest graded scale defensible from data alone, without
smuggling in an assumption about what attempt-count or solve-latency
*should* look like for a "good" recommendation — that's a product/ranking
philosophy decision already encoded in the heuristic ranker, not something
the label should re-litigate from a different, unstated assumption.

`BinarySolvedLabelStrategy` (0/1) is implemented alongside it for direct
comparison — pass `strategy=BinarySolvedLabelStrategy()` to `LabelGenerator`
to experiment with it against the same feature set.

## Configurable parameters

- `window_days` (default **14.0**, `training/config.py::DEFAULT_OUTCOME_WINDOW_DAYS`) —
  how long after `recommended_at` to look for a qualifying submission before
  calling it ignored. 14 days balances two failure modes: too short a
  window mislabels a genuinely-engaged-but-slow user as "ignored"; too long
  a window risks attributing an unrelated, much-later solve (e.g. the same
  problem surfacing again from a different pool weeks later) to *this*
  specific recommendation event. No historical outcome data exists yet to
  empirically tune this — 14 days is a reasoned default, not a fitted one,
  and is exposed as a constructor parameter specifically so it can be
  tuned once real outcome-latency data exists.
- `strategy` — swappable per `LabelStrategy` subclass; adding a new one
  requires no change to `LabelGenerator`, `DatasetGenerator`, or any
  recommender code (see "Adding a new strategy" below).

## Assumptions

- A submission with `submitted_at >= recommended_at` (not `>`) is
  attributed to this recommendation event — the boundary is inclusive
  since recommendation-then-immediate-submission is a real, valid case.
- Only `verdict == "Accepted"` counts as solved, matching the exact string
  `user_graph_service.py::_load_submissions` already checks — no new
  verdict taxonomy invented.
- A row's `label` is `NaN` (not 0) whenever its own `recommended_at`/
  `user_id`/`candidate_id` is missing — an unlabelable row is reported as
  unlabeled, never silently defaulted to "ignored."

## Expected impact on LambdaRank

Graded relevance (0/1/2) lets LambdaRank's pairwise loss learn three
distinct orderings per query group instead of one binary split — e.g., it
can learn that an attempted-but-failed candidate should still rank above a
never-shown-signal candidate of similar features, which a binary label
would treat identically. This is expected to produce better NDCG on
"do we rank engaging candidates above ignorable ones," even before any
solved outcomes exist, since attempted-vs-ignored data will be far more
abundant than solved-vs-not data early on (attempts happen well before most
recommended problems get solved).

## Adding a future strategy

1. Subclass `LabelStrategy`, implement `compute_label(outcome) -> float`,
   set `name` and `max_label`.
2. Register it in `STRATEGIES` (`training/label_generator.py`).
3. Nothing else changes — `LabelGenerator`, `DatasetGenerator`,
   `feature_extractor.py`, and the recommender itself are all unaffected.

## Known limitation and what would improve label quality most

The single biggest quality gap today: **no logged "recommendation was seen
but explicitly not engaged with" event** — see the `was_skipped` note
above. Every "ignored" label today is really "no submission arrived within
the window," which conflates genuine disinterest with "the user simply
hasn't gotten to it yet" or "was on a break from the platform entirely."
If frontend impression/dismiss events were logged to `recommendation_log`
(or a new table), the ignored/skipped distinction could become a real,
data-grounded fourth relevance tier instead of a time-window heuristic.

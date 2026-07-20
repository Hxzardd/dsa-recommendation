"""
postman/bulk_test_data/replay_100_events.py

Replays telemetry_100_events.json (10 synthetic cold-start users x 10
submissions each, increasing difficulty 0.196 -> 0.7) through the real
POST /update endpoint via TestClient, printing each user's mastery
progression per topic.

Source data: originally supplied as telemetry_100_events.json with
fabricated topic tags ("arrays", "strings", "dynamic-programming",
"graphs") that don't exist in data/problem_topic_edges_normalized.json,
and currentHlr as a bare float instead of the schema's expected dict.
Both fixed here at the source (not at replay time): tags mapped 1:1 to
their real manifest equivalents ("array", "string", "dynamic_programming",
"graph" -- verified valid for every specific problem they're used with),
currentHlr wrapped as {"half_life": <value>} (the float range, ~0.98-4.8,
matches HLR's expected half_life domain).

Requires a valid session token per user -- middlewares/auth.py requires a
Bearer token on every route, AND routes/submission.py cross-checks the
token's resolved user_id against submission.userId (403 on mismatch), so
ONE shared token across all 10 synthetic users does NOT work. This script
seeds a real user + session row (via seed_test_session.py's own functions,
same schema) for every distinct userId found in the dataset automatically,
using a short-lived 1-day token since these are throwaway test identities.

This WRITES to the real Postgres DB (same one seed_test_session.py writes
to) -- 10 user rows + 10 session rows, all under coldstart_user_XX ids.

Run:
    python postman/bulk_test_data/replay_100_events.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import main
from fastapi.testclient import TestClient
from seed_test_session import seed_user, seed_session

DATA_PATH = Path(__file__).parent / "telemetry_100_events.json"


def main_() -> None:
    client = TestClient(main.app)
    events = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    users = {}
    for e in events:
        users.setdefault(e["userId"], []).append(e)

    print(f"Seeding {len(users)} test users + session tokens (1-day validity)...")
    tokens = {}
    for user_id in users:
        seed_user(user_id, cf_handle=None, lc_handle=None)
        tokens[user_id] = seed_session(user_id, days_valid=1)
    print("Done seeding.\n")

    print("=" * 78)
    print(f"  REPLAYING {len(events)} EVENTS ACROSS {len(users)} USERS")
    print("=" * 78)

    for user_id, user_events in users.items():
        auth_headers = {"Authorization": f"Bearer {tokens[user_id]}"}
        print(f"\n--- {user_id} ({len(user_events)} submissions) ---")
        for e in user_events:
            r = client.post("/update", json=e, headers=auth_headers)
            if r.status_code != 200:
                print(f"    [FAILED {r.status_code}] {e['problemId']}: {r.json()}")
                continue
            body = r.json()
            # only print the topic the sample data explicitly seeded --
            # the other manifest-derived topics for the same problem also
            # update, but weren't part of this dataset's intent
            seeded_topics = {t["topicId"] for t in e.get("problemTopics", [])}
            shown = [t for t in body["updatedTopics"] if t["topicId"] in seeded_topics]
            summary = ", ".join(f"{t['topicId']}={t['updatedMastery']}" for t in shown)
            print(f"    {e['problemId']:45s} difficulty={e['problemDifficulty']:.3f} "
                  f"verdict={e['verdict']:20s} -> {summary}")

    print("\n" + "=" * 78)
    print("  Done.")
    print("=" * 78)


if __name__ == "__main__":
    main_()

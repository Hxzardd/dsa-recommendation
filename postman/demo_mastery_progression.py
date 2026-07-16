"""
postman/demo_mastery_progression.py

Demonstrates GET /mastery/{user_id} reflecting a user's actual progress
after their first few submissions.

POST /update is intentionally stateless (see controllers/submission_controller.py
docstring: "ML never touches the database -- backend owns persistence") -- it
computes and returns the new mastery/HLR values but never writes them anywhere.
In production, the backend is responsible for persisting each /update response
into Postgres. This script plays that backend role: call /update, persist the
returned updatedTopics into user_topic_mastery/user_hlr_state, then call
GET /mastery to show the change.

Run (needs Postgres reachable via DATABASE_URL, no Qdrant/Neo4j needed):
    python postman/demo_mastery_progression.py [user_id]
"""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, ".")

from main import app
import seed_test_session as sts
from database.postgres.db import get_connection, release_connection, _ml_slug_to_topic_id


def _persist_update_response(user_id: str, update_response: dict) -> None:
    """Backend-side responsibility: write /update's returned mastery into
    Postgres so GET /mastery picks it up, using the real, shared
    ML-slug -> topic.id translation (database/postgres/db.py's
    save_user_mastery, backed by database/postgres/topic_taxonomy.py) --
    not a demo-local guess. Overwrites on conflict (unlike seeding's
    DO NOTHING) since this simulates ongoing progress, not one-time import.
    user_hlr_state.p_recall is NOT NULL with no default and isn't part of
    /update's response shape, so HLR persistence is out of scope here."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for t in update_response["updatedTopics"]:
                topic_id = _ml_slug_to_topic_id(conn, t["topicId"])
                if topic_id is None:
                    print(f"    [skip persist] no backend topic for ML slug "
                          f"{t['topicId']!r} (see topic_taxonomy.UNMAPPED_TOPIC_SLUGS)")
                    continue
                cur.execute(
                    """
                    INSERT INTO user_topic_mastery (user_id, topic_id, mastery_score, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, topic_id) DO UPDATE SET
                        mastery_score = EXCLUDED.mastery_score,
                        updated_at = NOW()
                    """,
                    (user_id, topic_id, t["updatedMastery"]),
                )
        conn.commit()
    finally:
        release_connection(conn)


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else "mastery_progression_demo_user"

    sts.seed_user(user_id, None, None)
    token = sts.seed_session(user_id, 1)
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app)

    print(f"=== user: {user_id} ===\n")

    print("--- GET /mastery (before any submissions) ---")
    r = client.get(f"/mastery/{user_id}", headers=headers)
    print(json.dumps(r.json(), indent=2), "\n")

    submissions = [
        {
            "userId": user_id, "problemId": "demo_array_easy", "verdict": "OK", "score": 0.9,
            "problemTopics": [{"topicId": "array"}], "problemDifficulty": 0.2,
        },
        {
            "userId": user_id, "problemId": "demo_string_easy", "verdict": "OK", "score": 0.85,
            "problemTopics": [{"topicId": "string"}], "problemDifficulty": 0.25,
        },
        {
            "userId": user_id, "problemId": "demo_array_medium", "verdict": "OK", "score": 1.0,
            "problemTopics": [{"topicId": "array"}], "problemDifficulty": 0.45,
        },
    ]

    for i, sub in enumerate(submissions, 1):
        r = client.post("/update", json=sub, headers=headers)
        r.raise_for_status()
        update_response = r.json()
        _persist_update_response(user_id, update_response)

        print(f"--- submission {i}: {sub['problemId']} (verdict={sub['verdict']}, "
              f"difficulty={sub['problemDifficulty']}) ---")
        for t in update_response["updatedTopics"]:
            print(f"    {t['topicId']}: -> {t['updatedMastery']}")

        r = client.get(f"/mastery/{user_id}", headers=headers)
        print(f"--- GET /mastery after submission {i} ---")
        print(json.dumps(r.json(), indent=2), "\n")


if __name__ == "__main__":
    main()

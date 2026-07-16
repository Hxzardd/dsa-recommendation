"""
Create (or refresh) a throwaway test user for Postman testing against a
local Postgres seeded with database/postgres/dev_schema.sql.

At this commit, POST /update, GET /mastery, GET /urgency, and
GET /recommend have NO auth at all -- call them directly, no token needed.
Only POST /seed_hlr/{user_id} and POST /seed_bkt/{user_id} check anything,
and it's a documented placeholder (routes/seeding.py's require_same_user):
an `X-User-Id` header that must simply match the {user_id} path segment,
with no real signature/session verification behind it.

This script just inserts a `user` row (required by seeding_controller.py's
user_exists() check) and optionally sets linked_codeforces/linked_leetocde
so you can exercise the seeding routes too.

Usage:
    python seed_test_session.py                              # user_id=postman_demo_user
    python seed_test_session.py my_user_id
    python seed_test_session.py my_user_id --cf my_cf_handle --lc my_lc_handle
"""

import argparse

from database.postgres.db import get_connection


def seed_user(user_id: str, cf_handle: str | None, lc_handle: str | None) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO "user" (id, linked_codeforces, linked_leetcode)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    linked_codeforces = EXCLUDED.linked_codeforces,
                    linked_leetcode = EXCLUDED.linked_leetcode
                """,
                (user_id, cf_handle, lc_handle),
            )
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id", nargs="?", default="postman_demo_user")
    parser.add_argument("--cf", dest="cf_handle", default=None, help="Codeforces handle to link")
    parser.add_argument("--lc", dest="lc_handle", default=None, help="LeetCode handle to link")
    args = parser.parse_args()

    seed_user(args.user_id, args.cf_handle, args.lc_handle)

    print(f"\nSeeded user_id={args.user_id!r}"
          + (f" (codeforces={args.cf_handle!r})" if args.cf_handle else "")
          + (f" (leetcode={args.lc_handle!r})" if args.lc_handle else "") + "\n")
    print("Postman collection variables to set (or import postman_environment.json):")
    print(f"  user_id = {args.user_id}")
    print(f"\nCurl smoke tests (no auth needed for these):")
    print(f'  curl http://localhost:8000/mastery/{args.user_id}')
    print(f'  curl -X POST http://localhost:8000/update -H "Content-Type: application/json" '
          f'-d @postman/telemetry_samples/01_cold_start_easy_solve.json')
    print(f"\nSeeding routes DO check X-User-Id (placeholder, must equal the path user_id):")
    print(f'  curl -X POST http://localhost:8000/seed_bkt/{args.user_id} -H "X-User-Id: {args.user_id}"')


if __name__ == "__main__":
    main()

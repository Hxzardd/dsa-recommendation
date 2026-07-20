"""
Create (or refresh) a throwaway test user AND a real session token for
Postman/curl testing against a local Postgres seeded with
database/postgres/dev_schema.sql.

middlewares/auth.py now requires a valid `Authorization: Bearer <token>`
header resolving to a non-expired row in `session` on EVERY route except
"/", "/docs", "/openapi.json", "/redoc" -- there is no more
unauthenticated path for POST /update, GET /mastery, GET /urgency, or
GET /recommend. This script mints exactly that: a `user` row plus a
`session` row with a fresh token, expiring 30 days out.

POST /seed_hlr/{user_id} and POST /seed_bkt/{user_id} separately ALSO
check X-User-Id (routes/seeding.py's require_same_user, a documented
placeholder equality check) -- that's on top of, not instead of, the
Bearer token this script mints.

Usage:
    python seed_test_session.py                              # user_id=postman_demo_user
    python seed_test_session.py my_user_id
    python seed_test_session.py my_user_id --cf my_cf_handle --lc my_lc_handle
    python seed_test_session.py my_user_id --days 7           # shorter-lived token
"""

import argparse
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from database.postgres.db import get_connection, release_connection


def seed_user(user_id: str, cf_handle: str | None, lc_handle: str | None) -> None:
    # email is NOT NULL with no default in the real schema (verified
    # directly: only id/email are required without a default) -- synthesize
    # one from user_id since this is a throwaway test user, not a real signup.
    email = f"{user_id}@test.local"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO "user" (id, email, linked_codeforces, linked_leetcode)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    linked_codeforces = EXCLUDED.linked_codeforces,
                    linked_leetcode = EXCLUDED.linked_leetcode
                """,
                (user_id, email, cf_handle, lc_handle),
            )
        conn.commit()
    finally:
        # database/postgres/db.py's get_connection() now draws from a
        # ThreadedConnectionPool (maxconn=10) -- conn.close() closes the
        # TCP connection but does NOT tell the pool it's free, so the pool
        # still counts it as checked out. release_connection() (putconn())
        # is the only way to actually return a slot to the pool; calling
        # .close() here would silently leak one pool slot per call until
        # the pool exhausts itself (confirmed: this bit us seeding 10 test
        # users in a loop -- 20 get_connection() calls with no release hit
        # "connection pool exhausted" well before reaching maxconn=10 x 2).
        release_connection(conn)


def seed_session(user_id: str, days_valid: int) -> str:
    """Creates a fresh session row and returns its bearer token. Any prior
    sessions for this user are left alone (multiple valid sessions per user
    is normal) -- this always mints a NEW token rather than trying to find
    and reuse an existing unexpired one, so repeated runs are predictable."""
    token = secrets.token_urlsafe(32)
    session_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(days=days_valid)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session (id, user_id, token, expires_at)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, user_id, token, expires_at),
            )
        conn.commit()
    finally:
        release_connection(conn)
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id", nargs="?", default="postman_demo_user")
    parser.add_argument("--cf", dest="cf_handle", default=None, help="Codeforces handle to link")
    parser.add_argument("--lc", dest="lc_handle", default=None, help="LeetCode handle to link")
    parser.add_argument("--days", type=int, default=30, help="Session validity in days (default 30)")
    args = parser.parse_args()

    seed_user(args.user_id, args.cf_handle, args.lc_handle)
    token = seed_session(args.user_id, args.days)

    print(f"\nSeeded user_id={args.user_id!r}"
          + (f" (codeforces={args.cf_handle!r})" if args.cf_handle else "")
          + (f" (leetcode={args.lc_handle!r})" if args.lc_handle else "") + "\n")
    print(f"Session token (valid {args.days} days): {token}\n")
    print("Postman collection variables to set (or import postman_environment.json):")
    print(f"  user_id = {args.user_id}")
    print(f"  auth_token = {token}")
    print(f"\nCurl smoke tests (Authorization header now required on every route):")
    print(f'  curl http://localhost:8000/mastery/{args.user_id} '
          f'-H "Authorization: Bearer {token}"')
    print(f'  curl -X POST http://localhost:8000/update -H "Content-Type: application/json" '
          f'-H "Authorization: Bearer {token}" -d @postman/telemetry_samples/01_cold_start_easy_solve.json')
    print(f"\nSeeding routes check X-User-Id ON TOP OF the Bearer token "
          f"(placeholder, must equal the path user_id):")
    print(f'  curl -X POST http://localhost:8000/seed_bkt/{args.user_id} '
          f'-H "Authorization: Bearer {token}" -H "X-User-Id: {args.user_id}"')


if __name__ == "__main__":
    main()

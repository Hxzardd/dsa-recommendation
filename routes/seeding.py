"""
Seeding routes.

FIX (Greptile P1 "Cross-User Seeding Is Unrestricted"): these routes
previously accepted ANY path-supplied user_id with no check that the
caller actually IS that user. Anyone could trigger CF/LC history reads
and persistent mastery/HLR writes for an arbitrary other user_id, and
the different response messages ("No handle linked" vs actual seeding
results) also leaked whether that user has a linked account -- an
information disclosure on top of the write authorization gap.

require_same_user() below rejects (403) if the caller's identity
doesn't match the path user_id. get_current_user_id() below is a
PLACEHOLDER that reads an X-User-Id header as a stand-in -- that alone
is NOT secure (any client can set any header value). It MUST be
replaced with real session/JWT verification before this is safe to
deploy. It exists only to keep the dependency wiring structurally
correct while real auth gets plugged in.
"""

from fastapi import APIRouter, Header, HTTPException
from controllers.seeding_controller import handle_seed_hlr, handle_seed_bkt

router = APIRouter()


def require_same_user(path_user_id: str, caller_user_id: str) -> None:
    """
    Raises 403 if the caller isn't the user_id they're trying to seed
    data for.

    caller_user_id currently comes from an X-User-Id header (see the
    route functions below) -- a PLACEHOLDER identity source. Replace
    with real session/JWT verification, e.g.:

        session_token = request.cookies.get("session")
        session = verify_session(session_token)   # your Better Auth / JWT check
        if session is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        caller_user_id = session.user_id
    """
    if not caller_user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if path_user_id != caller_user_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot seed data for another user",
        )


@router.post("/seed_hlr/{user_id}")
def seed_hlr(user_id: str, caller_user_id: str = Header(None, alias="X-User-Id")):
    require_same_user(user_id, caller_user_id)
    return handle_seed_hlr(user_id)


@router.post("/seed_bkt/{user_id}")
def seed_bkt(user_id: str, caller_user_id: str = Header(None, alias="X-User-Id")):
    require_same_user(user_id, caller_user_id)
    return handle_seed_bkt(user_id)
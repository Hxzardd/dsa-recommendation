"""
tests/test_auth_middleware.py

Tests middlewares/auth.py's ML_SERVICE_TOKEN bypass -- server-to-server
calls (backend's judge0 submission webhook) with no end-user session.

Run:
    python -m pytest tests/test_auth_middleware.py -v
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from starlette.requests import Request

from middlewares import auth as auth_module


def _make_request(authorization: str | None, path: str = "/update") -> Request:
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
    }
    return Request(scope)


async def _noop_call_next(request):
    return "OK"


class TestServiceTokenBypass(unittest.TestCase):

    def setUp(self):
        self._orig_token = auth_module._ML_SERVICE_TOKEN

    def tearDown(self):
        auth_module._ML_SERVICE_TOKEN = self._orig_token

    def test_matching_service_token_bypasses_session_lookup(self):
        auth_module._ML_SERVICE_TOKEN = "shared-secret-123"
        request = _make_request("Bearer shared-secret-123")

        with patch.object(auth_module, "verify_session_token") as mock_verify:
            result = asyncio.run(auth_module.auth_middleware(request, _noop_call_next))
            mock_verify.assert_not_called()

        self.assertEqual(result, "OK")
        self.assertTrue(request.state.is_service_call)
        self.assertIsNone(request.state.user_id)

    def test_non_matching_token_falls_through_to_session_lookup(self):
        auth_module._ML_SERVICE_TOKEN = "shared-secret-123"
        request = _make_request("Bearer some-other-token")

        with patch.object(auth_module, "verify_session_token", return_value="user_1") as mock_verify:
            result = asyncio.run(auth_module.auth_middleware(request, _noop_call_next))
            mock_verify.assert_called_once_with("some-other-token")

        self.assertEqual(result, "OK")
        self.assertFalse(request.state.is_service_call)
        self.assertEqual(request.state.user_id, "user_1")

    def test_service_token_disabled_when_unset(self):
        auth_module._ML_SERVICE_TOKEN = None
        request = _make_request("Bearer whatever-value")

        with patch.object(auth_module, "verify_session_token", return_value="user_1") as mock_verify:
            asyncio.run(auth_module.auth_middleware(request, _noop_call_next))
            mock_verify.assert_called_once_with("whatever-value")

        self.assertFalse(request.state.is_service_call)

    def test_is_service_token_helper_matches_and_rejects(self):
        auth_module._ML_SERVICE_TOKEN = "abc123"
        self.assertTrue(auth_module._is_service_token("abc123"))
        self.assertFalse(auth_module._is_service_token("wrong"))
        self.assertFalse(auth_module._is_service_token(""))

        auth_module._ML_SERVICE_TOKEN = None
        self.assertFalse(auth_module._is_service_token("abc123"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

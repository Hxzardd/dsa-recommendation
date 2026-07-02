"""
Backward-compatible application entrypoint.

Deprecated:
    Use app.main:app instead.
"""

from app.main import app

__all__ = ["app"]
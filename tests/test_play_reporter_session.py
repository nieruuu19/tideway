"""JWT session-id extraction used by the realtime self-event filter.

Tidal echoes a client's own playback back as a
PRIVILEGED_SESSION_NOTIFICATION; the listener compares the frame's
`sessionId` with our token's `sid`. If extraction silently returned
something else, Tideway would either pause on every play (filter
never matches) or ignore a real other-device event (filter matches
everything).
"""
from __future__ import annotations

import base64
import json

from app.play_reporter import session_id_from_token


def _token(claims: dict) -> str:
    def seg(obj) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'HS256', 'typ': 'JWT'})}.{seg(claims)}.sig"


def test_returns_sid_claim():
    token = _token({"sid": "0d8fecb2-dcdb-4555-97b0-7c807806e4ad", "cid": 8017})
    assert (
        session_id_from_token(token)
        == "0d8fecb2-dcdb-4555-97b0-7c807806e4ad"
    )


def test_missing_sid_is_none():
    assert session_id_from_token(_token({"cid": 8017})) is None


def test_empty_or_none_token_is_none():
    assert session_id_from_token(None) is None
    assert session_id_from_token("") is None


def test_malformed_token_is_none():
    assert session_id_from_token("not-a-jwt") is None
    assert session_id_from_token("a.b") is None
    assert session_id_from_token("a.!!!.c") is None

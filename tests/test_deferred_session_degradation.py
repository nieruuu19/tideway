"""A deferred Tidal session must degrade, not blow up.

`load_session()` cannot always validate the stored session at boot — no
network yet, or (as in the 2026-10-28 log) a 429 from Tidal while the
app is still starting. The credentials remain valid and
`_session_load_deferred` flips on, which is what makes `/api/auth/status`
answer "signed in, offline" so the local library still works. But
`session.user` is still None, so helpers that dereference it directly
crashed with AttributeError instead of returning the same degraded
answer the rest of the offline path gives:

    File "app/tidal_client.py", line 1840, in get_user_playlists
        items = _fetch_all_pages(self.session.user.playlists)
    AttributeError: 'NoneType' object has no attribute 'playlists'

That surfaced as `/api/playlists/mine` -> 500. Two more handlers showed
the wrong thing during a backoff window: `/api/page/{name}` turned the
refusal into a 502 and `/api/album/{id}` into a 404, so the album page
claimed the record was gone. Both now re-raise TidalBackoffError so the
shared handler can answer 503 + Retry-After and the backoff banner can
explain the wait.
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.audio.player import PCMPlayer
from app.tidal_client import TidalBackoffError, TidalClient


def _client_without_user() -> TidalClient:
    """A client in the exact state a deferred boot leaves behind."""
    client = object.__new__(TidalClient)
    client.session = SimpleNamespace(user=None)
    return client


def _refuse(*_args, **_kwargs):
    raise TidalBackoffError(42, "rate-limited (HTTP 429)")


def test_user_playlists_are_empty_before_the_session_loads():
    assert _client_without_user().get_user_playlists() == []


def test_favorite_explains_the_pending_session():
    with pytest.raises(RuntimeError, match="session isn't ready"):
        _client_without_user().favorite("track", "123", add=True)


def test_create_playlist_explains_the_pending_session():
    with pytest.raises(RuntimeError, match="session isn't ready"):
        _client_without_user().create_playlist("new playlist")


def test_page_endpoint_lets_a_backoff_answer_503(monkeypatch):
    import server

    server._invalidate_page_cache()
    monkeypatch.setattr(server, "_require_auth", lambda: None)
    monkeypatch.setitem(server._KNOWN_PAGES, "home", _refuse)

    with pytest.raises(TidalBackoffError):
        server.editorial_page("home")


def test_album_endpoint_lets_a_backoff_answer_503(monkeypatch):
    import server

    server._invalidate_detail_cache()
    monkeypatch.setattr(server, "_require_auth", lambda: None)
    monkeypatch.setattr(server.tidal.session, "album", _refuse)

    with pytest.raises(TidalBackoffError):
        server.album_detail(293931)


def test_artist_endpoint_lets_a_backoff_answer_503(monkeypatch):
    import server

    server._invalidate_artist_detail_cache()
    monkeypatch.setattr(server, "_require_auth", lambda: None)
    monkeypatch.setattr(server.tidal.session, "artist", _refuse)

    with pytest.raises(TidalBackoffError):
        server.artist_detail(15574)


def test_radios_let_a_backoff_answer_503(monkeypatch):
    import server

    monkeypatch.setattr(server, "_require_auth", lambda: None)
    monkeypatch.setattr(server.tidal.session, "track", _refuse)
    monkeypatch.setattr(server.tidal.session, "artist", _refuse)

    with pytest.raises(TidalBackoffError):
        server.track_radio(293944)
    with pytest.raises(TidalBackoffError):
        server.artist_radio(15574)


# ---------------------------------------------------------------------------
# Player snapshot — the frontend skips past unplayable tracks (region
# locks, etc.) by auto-advancing on state=error. A rate-limit refusal
# isn't about the track, so the snapshot has to say so or the queue
# walks itself into the same closed gate track after track.
# ---------------------------------------------------------------------------


def _errored_player(message: str) -> PCMPlayer:
    player = PCMPlayer(session_getter=lambda: MagicMock())
    with player._lock:
        player._state = "error"
        player._last_error = message
    return player


def test_snapshot_marks_a_rate_limit_error(monkeypatch):
    from app import tidal_client

    monkeypatch.setattr(tidal_client, "_tidal_backoff_until", time.time() + 30)
    snap = _errored_player("TIDAL is rate-limiting requests").snapshot()
    assert snap.error_kind == "backoff"


def test_snapshot_leaves_ordinary_errors_skippable():
    snap = _errored_player("manifest has no segment urls").snapshot()
    assert snap.error_kind is None


def test_the_api_payload_carries_the_error_kind(monkeypatch):
    """The frontend only ever sees `_snapshot_dict` output — over SSE
    and over every HTTP player response. A field that stops at the
    dataclass is dead weight: the client then can't tell a rate-limit
    refusal from a broken row, skips, and walks the queue back into the
    closed gate (which is what tripped a 60 s suspension while an album
    was being clicked through)."""
    import server
    from app import tidal_client

    monkeypatch.setattr(tidal_client, "_tidal_backoff_until", time.time() + 30)
    payload = server._snapshot_dict(
        _errored_player("TIDAL is rate-limiting requests").snapshot()
    )
    assert payload["error_kind"] == "backoff"


def test_the_api_payload_leaves_ordinary_errors_skippable():
    import server

    payload = server._snapshot_dict(
        _errored_player("manifest has no segment urls").snapshot()
    )
    assert payload["error_kind"] is None

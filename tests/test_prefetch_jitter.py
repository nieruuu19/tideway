"""Tests for the album-mount prefetch fan-out fix.

The prefetch endpoint runs N tracks through a ThreadPoolExecutor; each
track triggers two parallel Tidal API calls inside `_resolve_source`
(track metadata + playbackinfo) followed by a local manifest parse.
Without per-worker jitter, all N workers fire their first call within
a few ms of each other and the resulting burst is exactly the shape
Tidal's anti-abuse layer reads as "client is scraping": first 429,
then 403/`abuse_detected`, then the longer escalation. Putting
`tidal_jitter_sleep()` at the top of `player.prefetch` spreads the
lead-in across a 50-200 ms window per worker, so the first request
from each worker lands at a different moment before the parallel
pair fires.

These tests pin both halves of that contract:
- `prefetch` calls `tidal_jitter_sleep` before resolving (so
  refactors can't silently remove it).
- The prefetch endpoint warms tracks serially (so an album mount can
  never outrun the request rate playback itself produces), and it
  no-ops while the Tidal gate is closed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_player_prefetch_jitters_before_first_tidal_call():
    """The lazy import inside `prefetch` resolves
    `tidal_jitter_sleep` from `app.tidal_client` at call time. Patch
    it there and the call should land before _resolve_source runs.
    """
    from app.audio.player import PCMPlayer

    player = PCMPlayer.__new__(PCMPlayer)  # bypass __init__ for unit isolation
    player._manifest_cache = MagicMock()
    player._warm_bytes = MagicMock()

    call_order: list[str] = []

    def fake_jitter() -> None:
        call_order.append("jitter")

    def fake_resolve(track_id, quality, *, set_current_meta=True):
        call_order.append("resolve")
        # Mimic _resolve_source's return shape:
        # (urls, dur, info, bytes_map, track_meta).
        return (["http://seg/0", "http://seg/1"], 180.0, None, {}, None)

    player._resolve_source = fake_resolve  # type: ignore[assignment]

    with patch("app.tidal_client.tidal_jitter_sleep", side_effect=fake_jitter):
        ok = player.prefetch("12345", quality=None, warm_bytes=False)

    assert ok is True
    # Jitter must come before resolve, not after. The whole point is
    # to spread the lead-in to Tidal, so a post-resolve jitter would
    # be useless.
    assert call_order == ["jitter", "resolve"], call_order


def test_prefetch_endpoint_warms_one_track_at_a_time(monkeypatch):
    """Ten tracks at once is the shape that got the account suspended:
    the per-track resolve fans out to several Tidal calls, so two
    workers held the request rate at several per second. Warming
    serially keeps it at the rate playback itself produces, and the
    album's first tracks are still ready first because the order is
    preserved."""
    import threading
    import time

    import server
    from app import tidal_client

    monkeypatch.setattr(tidal_client, "_tidal_backoff_until", 0.0)
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()
    order: list[str] = []

    def fake_prefetch(track_id, quality, *, warm_bytes=True):  # noqa: ARG001
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            order.append(track_id)
        time.sleep(0.01)
        with lock:
            state["now"] -= 1
        return True

    fake_player = MagicMock()
    fake_player.prefetch = fake_prefetch
    monkeypatch.setattr(server, "_native_player", lambda: fake_player)
    monkeypatch.setattr(server, "_require_local_access", lambda: None)
    monkeypatch.setattr(server.settings, "offline_mode", False)

    ids = [str(i) for i in range(10)]
    out = server.player_prefetch(
        server._PlayerPrefetchRequest(track_ids=ids, quality=None)
    )

    assert state["peak"] == 1, f"peak concurrent prefetch was {state['peak']}"
    assert order == ids
    assert out == {"prefetched": 10, "total": 10}


def test_prefetch_endpoint_skips_while_the_gate_is_closed(monkeypatch):
    """A suspended account refuses every resolve locally, so the pool
    would only park workers in jitter sleeps — and album mounts land
    in exactly this window (the user browses while playback is
    blocked)."""
    import time

    import server
    from app import tidal_client

    monkeypatch.setattr(tidal_client, "_tidal_backoff_until", time.time() + 30)
    fake_player = MagicMock()
    monkeypatch.setattr(server, "_native_player", lambda: fake_player)
    monkeypatch.setattr(server, "_require_local_access", lambda: None)
    monkeypatch.setattr(server.settings, "offline_mode", False)

    out = server.player_prefetch(
        server._PlayerPrefetchRequest(track_ids=["1", "2"], quality=None)
    )

    assert out["skipped"] == "backoff"
    assert out["prefetched"] == 0
    fake_player.prefetch.assert_not_called()

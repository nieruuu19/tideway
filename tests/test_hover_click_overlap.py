"""A click that lands on top of a hover prefetch.

The frontend fires a prefetch on hover and the click follows a few
hundred ms later, while that prefetch is still resolving or still
downloading the init segment. Two things made that overlap cost more
than doing nothing: both sides resolved the same track (six Tidal
calls instead of three, the click paying the full round-trip), and
both sides downloaded the same 400 KB first segment for the DASH
prime. These tests pin the sharing contract.
"""
from __future__ import annotations

import threading
import time

from app.audio.manifest_cache import ManifestCache


def _player():
    from app.audio.player import PCMPlayer

    player = PCMPlayer.__new__(PCMPlayer)
    player._manifest_cache = ManifestCache()
    player._local_lookup = None
    player._quality_clamp = None
    player._resolving = {}
    player._resolving_lock = threading.Lock()
    player._warm_inflight = {}
    player._warm_lock = threading.Lock()
    player._current_track_meta = None
    return player


def _spec(meta):
    urls = ["http://seg/0", "http://seg/1"]
    return (urls, 180.0, object(), {}, meta)


def test_a_click_joins_the_resolve_a_hover_prefetch_started():
    player = _player()
    calls: list[bool] = []
    started = threading.Event()

    def slow_resolve(_session, _track_id, _quality, _key, _t0, *, set_current_meta=True):
        calls.append(set_current_meta)
        started.set()
        time.sleep(0.4)
        return _spec({"title": "Mack The Knife"})

    player._session_getter = lambda: object()
    player._resolve_uncached = slow_resolve  # type: ignore[assignment]

    results: dict[str, tuple] = {}

    def caller(name, set_meta):
        results[name] = player._resolve_source(
            "12345", "atmos", set_current_meta=set_meta
        )

    hover = threading.Thread(target=caller, args=("hover", False))
    hover.start()
    assert started.wait(2.0)
    click = threading.Thread(target=caller, args=("click", True))
    click.start()
    hover.join(5.0)
    click.join(5.0)

    # One resolve served both callers.
    assert calls == [False], calls
    assert results["click"][0] == results["hover"][0]
    # The hover resolved speculatively; the click still publishes the
    # metadata the renderer reads.
    assert player._current_track_meta == {"title": "Mack The Knife"}


def test_a_joined_call_retries_after_a_failed_resolve():
    player = _player()
    calls: list[int] = []
    started = threading.Event()

    def flaky_resolve(*_args, **_kwargs):
        calls.append(1)
        started.set()
        time.sleep(0.2)
        if len(calls) == 1:
            raise RuntimeError("Tidal is down")
        return _spec({"title": "Recovered"})

    player._session_getter = lambda: object()
    player._resolve_uncached = flaky_resolve  # type: ignore[assignment]

    errors: dict[str, Exception] = {}
    results: dict[str, tuple] = {}

    def caller(name):
        try:
            results[name] = player._resolve_source("999", "atmos")
        except Exception as exc:  # noqa: BLE001 - the test asserts on it
            errors[name] = exc

    hover = threading.Thread(target=caller, args=("hover",))
    hover.start()
    assert started.wait(2.0)
    click = threading.Thread(target=caller, args=("click",))
    click.start()
    hover.join(5.0)
    click.join(5.0)

    assert calls == [1, 1], "the click retries after the hover's failure"
    assert isinstance(errors.get("hover"), RuntimeError)
    assert results["click"][4] == {"title": "Recovered"}
    # The registry is clean again, so the next click starts fresh.
    assert player._resolving == {}


def test_in_flight_prefetch_bytes_are_adopted_instead_of_downloaded_again():
    player = _player()
    key = ("12345", "atmos")
    urls = ["http://seg/0", "http://seg/1"]
    player._manifest_cache.store(key, list(urls), 180.0, object())

    requests: list[str] = []
    started = threading.Event()

    class _Response:
        def __init__(self, body: bytes) -> None:
            self.content = body

        def raise_for_status(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_get(url, timeout=None, headers=None, stream=False):
        # noqa: ARG001 - mirrors STREAM_SESSION.get's call signature
        requests.append(url)
        started.set()
        time.sleep(0.25)
        return _Response(b"init" if url.endswith("0") else b"seg1")

    import app.audio.player as player_mod

    original = player_mod.STREAM_SESSION.get
    player_mod.STREAM_SESSION.get = fake_get
    try:
        warm = threading.Thread(
            target=lambda: player._warm_bytes(key, urls), daemon=True
        )
        warm.start()
        assert started.wait(2.0)
        deadline = time.monotonic() + 2.0
        while len(player._warm_inflight.get(key, {})) < 2:
            assert time.monotonic() < deadline, "warm fetches never registered"
            time.sleep(0.01)

        merged = player._adopt_warm_bytes(key, None)
        warm.join(5.0)
    finally:
        player_mod.STREAM_SESSION.get = original

    assert merged == {0: b"init", 1: b"seg1"}
    # Each segment was pulled exactly once, by the prefetch.
    assert requests == urls
    # And the bytes the click took are the ones now cached.
    cached = player._manifest_cache.lookup(key)
    assert cached is not None and cached[3] == {0: b"init", 1: b"seg1"}
    assert player._warm_inflight == {}

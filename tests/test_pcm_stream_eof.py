"""Regression guard for the dead-stream-after-EOF freeze.

When a track plays to natural end-of-file, the audio callback raises
`sd.CallbackStop`. PortAudio's docs say the stream is now in a
"stopped" terminal state and `Pa_StartStream` cannot revive it — the
stream must be reopened. Before this fix, `_on_stream_finished`
transitioned state to "ended" but left `self._stream` pointing at the
corpse. The next `load()` then snapshotted that ref as `active_stream`,
took path (A) ("kept stream open"), and rebound the new pipeline
against a dead OutputStream. `play()` called `stream.start()`, which
on macOS returns without error but never re-wakes the CoreAudio
callback thread. State said "playing", no callback fired, the
decoder filled the queue and blocked.

This test pins the contract: on natural EOF, the stream ref is
cleared AND the underlying OutputStream is closed.
"""
from __future__ import annotations

import queue
import threading

from app.audio.player import PCMPlayer


def _player() -> PCMPlayer:
    return PCMPlayer(lambda: None)


class _FakeStream:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_natural_eof_closes_and_nulls_stream():
    p = _player()
    fake = _FakeStream()
    p._stream = fake
    p._state = "playing"
    p._replacing_stream = False
    p._preload = None
    p._pcm_queue = queue.Queue()
    p._decoder_done = threading.Event()
    p._decoder_done.set()

    p._on_stream_finished()

    assert p._stream is None, (
        "natural EOF must clear self._stream so next load() takes "
        "the full-reopen path, not the kept-stream-open path"
    )
    assert fake.closed, "the dead OutputStream must be closed"
    assert p._state == "ended"


def test_eof_close_swallows_close_errors():
    """A close() that raises must not propagate — finished_callback
    runs on sounddevice's own thread and a bare exception there
    would crash the audio path entirely. The state transition still
    has to happen."""
    p = _player()

    class _RaisingStream:
        def close(self) -> None:
            raise RuntimeError("boom")

    p._stream = _RaisingStream()
    p._state = "playing"
    p._replacing_stream = False
    p._preload = None
    p._pcm_queue = queue.Queue()
    p._decoder_done = threading.Event()
    p._decoder_done.set()

    p._on_stream_finished()

    assert p._stream is None
    assert p._state == "ended"


def test_device_loss_path_does_not_null_stream():
    """The device-loss recovery path needs `self._stream` to still
    point at the (just-aborted) OutputStream so it can be replaced.
    The EOF close-and-null behavior must only fire on natural EOF,
    not when the decoder still has work pending."""
    p = _player()
    fake = _FakeStream()
    p._stream = fake
    p._state = "playing"
    p._replacing_stream = False
    p._preload = None
    p._pcm_queue = queue.Queue()
    p._decoder_done = threading.Event()
    # decoder NOT done -> device-loss heuristic should fire and
    # spawn the recovery thread, leaving self._stream alone. Stub
    # recovery: the real one immediately takes ownership of
    # `self._stream`, which is exactly what this test is not about.
    p._recover_from_device_loss = lambda: None

    p._on_stream_finished()

    assert p._stream is fake, (
        "device-loss path must not null self._stream — recovery "
        "needs to reopen on a fresh device"
    )
    assert not fake.closed


class _NoopStream:
    """Structurally enough for `_on_stream_finished`; `active` is what
    the watchdog and the recovery paths look at."""

    def __init__(self, active: bool = True) -> None:
        self.active = active

    def close(self) -> None:
        pass


def _wait_recovery(p) -> bool:
    done = threading.Event()
    calls: list[int] = []

    def _recover() -> None:
        calls.append(1)
        done.set()

    p._recover_from_device_loss = _recover
    p._on_stream_finished()
    return done.wait(2.0) and bool(calls)


def test_paused_stream_ending_recovers_instead_of_ending():
    """Enabling Spatial Sound kills the stream while the user is
    paused. The callback never drains while paused, so this can't be
    end-of-track; emitting `ended` would make the frontend advance."""
    p = _player()
    p._stream = _NoopStream()
    p._state = "paused"
    p._replacing_stream = False
    p._preload = None
    p._pcm_queue = queue.Queue()
    p._decoder_done = threading.Event()

    assert _wait_recovery(p) is True

    assert p._state == "paused"


def test_device_loss_with_a_preload_armed_recovers_the_current_track():
    """A preload no longer exempts the stream from recovery: bridging
    to the next track on a dead endpoint just moved the failure one
    track later. The preload must survive for the real transition."""
    p = _player()
    p._stream = _NoopStream()
    p._state = "playing"
    p._replacing_stream = False
    pre = object()
    p._preload = pre
    p._pcm_queue = queue.Queue()
    p._decoder_done = threading.Event()

    assert _wait_recovery(p) is True

    assert p._preload is pre
    assert p._state == "playing"


def test_watchdog_recovers_a_stream_that_died_silently():
    p = _player()
    p._stream = _NoopStream(active=False)
    p._state = "playing"
    p._replacing_stream = False
    p._last_watchdog_recover_t = 0.0
    done = threading.Event()
    p._recover_from_device_loss = lambda: done.set()

    assert p._watchdog_recover_dead_stream(p._stream) is True
    assert done.wait(2.0)


def test_watchdog_leaves_a_live_stream_alone():
    p = _player()
    live = _NoopStream(active=True)
    p._replacing_stream = False
    spawned: list[int] = []
    p._recover_from_device_loss = lambda: spawned.append(1)

    assert p._watchdog_recover_dead_stream(live) is False
    assert spawned == []


def test_watchdog_rate_limits_recovery():
    p = _player()
    p._replacing_stream = False
    p._last_watchdog_recover_t = 0.0
    spawned: list[int] = []
    p._recover_from_device_loss = lambda: spawned.append(1)

    first = p._watchdog_recover_dead_stream(_NoopStream(active=False))
    second = p._watchdog_recover_dead_stream(_NoopStream(active=False))

    assert first is True
    assert second is False

"""Tests for the teardown races that crashed the backend.

Two crash dumps from the same session pinned native access violations
to two races:

  * ``libportaudio64bit+0xce38`` on ``pcm-device-recovery`` (three
    times) — sounddevice's ``OutputStream.close()`` is not atomic, so
    the finished callback and device recovery both closed the same
    handle. ``pa_guard`` serializes and de-duplicates that.
  * ``avformat-63+0xede29`` on ``pcm-preload`` — a dropped preload had
    its PyAV container freed while the decode thread was still inside
    libavformat. ``Decoder`` now serializes ``close()`` against
    ``next_pcm()``.

These tests pin the invariants with fakes; no audio device is opened.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.audio import pa_guard
from app.audio.decoder import Decoder


class _FakeStream:
    def __init__(self):
        self.closed = False
        self.aborts = 0
        self.closes = 0
        self._lock = threading.Lock()

    def abort(self):
        with self._lock:
            self.aborts += 1

    def close(self):
        with self._lock:
            self.closes += 1
            self.closed = True


def test_close_pa_stream_is_idempotent():
    stream = _FakeStream()
    pa_guard.close_stream(stream)
    pa_guard.close_stream(stream)
    pa_guard.close_stream(None)
    assert stream.closes == 1
    assert stream.aborts == 1


def test_concurrent_close_pa_stream_touches_the_stream_once():
    """The exact race from the crash dumps: two threads releasing the
    same handle. Only one may reach PortAudio."""
    stream = _FakeStream()
    start = threading.Barrier(8)

    def closer():
        start.wait()
        pa_guard.close_stream(stream)

    threads = [threading.Thread(target=closer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert stream.closes == 1
    assert stream.aborts == 1


def test_reinit_is_skipped_while_a_stream_is_open(monkeypatch):
    stream = _FakeStream()
    pa_guard._STREAMS.add(stream)
    calls = []
    monkeypatch.setattr(pa_guard.sd, "_terminate", lambda: calls.append("term"))
    monkeypatch.setattr(pa_guard.sd, "_initialize", lambda: calls.append("init"))
    try:
        assert pa_guard.reinit_portaudio() is False
        assert calls == []
        assert pa_guard.live_stream_count() == 1
        pa_guard.close_stream(stream)
        assert pa_guard.reinit_portaudio() is True
        assert calls == ["term", "init"]
    finally:
        pa_guard._STREAMS.discard(stream)


class _GatedContainer:
    """Minimal stand-in for a PyAV container whose decode blocks until
    the test releases it, i.e. the decode thread is inside libav."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = 0

    def decode(self, _stream):
        self.entered.set()
        self.release.wait(5)
        return
        yield  # pragma: no cover - unreachable, keeps this a generator

    def close(self):
        self.closed += 1


class _NullResampler:
    def resample(self, _frames):
        return []


def _bare_decoder(container) -> Decoder:
    decoder = object.__new__(Decoder)
    decoder._io_lock = threading.Lock()
    decoder._closed = False
    decoder._container = container
    decoder._source = None
    decoder._stream = object()
    decoder._iter = None
    decoder._done = False
    decoder._pending_seek_s = None
    decoder._resampler = _NullResampler()
    return decoder


def test_close_waits_for_the_decode_thread_to_leave_libav(monkeypatch):
    container = _GatedContainer()
    decoder = _bare_decoder(container)
    monkeypatch.setattr(Decoder, "_CLOSE_LOCK_WAIT_S", 0.05)

    decode_done = threading.Event()

    def decode():
        decoder.next_pcm()
        decode_done.set()

    t = threading.Thread(target=decode, daemon=True)
    t.start()
    assert container.entered.wait(5)

    decoder.close()
    # The decode thread still owns libav, so the container must survive.
    assert container.closed == 0
    assert decoder._closed is True

    container.release.set()
    assert decode_done.wait(5)
    t.join(5)

    # Once the thread is out, a later close frees it exactly once.
    decoder.close()
    assert container.closed == 1


def test_next_pcm_returns_none_after_close(monkeypatch):
    container = _GatedContainer()
    decoder = _bare_decoder(container)
    decoder.close()
    assert decoder.next_pcm() is None
    container.release.set()
    # Never entered libav at all, so nothing was decoded or closed twice.
    assert container.closed == 1
    assert not container.entered.is_set()


def test_decode_loop_exits_after_close():
    """The player's decoder loop polls next_pcm(); a closed decoder has
    to read as end-of-stream so the thread unwinds instead of spinning."""
    decoder = _bare_decoder(_GatedContainer())
    decoder._container.release.set()
    decoder.close()
    assert decoder.next_pcm() is None
    assert decoder.next_pcm() is None

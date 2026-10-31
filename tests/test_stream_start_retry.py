"""PCMPlayer._ensure_stream_started survives a lost WASAPI start race.

PortAudio's Windows backend fails IAudioClient::Start with an
`Unanticipated host error` when the driver is still releasing the
endpoint after close(); the same parameters start fine moments later.
The helper reopens and retries instead of parking the player in
`error`, and re-raises the last failure when every attempt fails.

A Windows Spatial Sound change is the harder version of the same
failure: the cached WASAPI endpoint stays unstartable until PortAudio
re-enumerates, and only another host-API copy of the same speakers may
accept the stream. The retries therefore widen — reopen, re-enumerate,
alternate host-API copy, system default — and a stream that never
started is dropped so the next track cannot reuse it.
"""
from __future__ import annotations

import logging
import threading

import pytest

from app.audio import player
from app.audio.player import PCMPlayer


class FakeStream:
    def __init__(self, failures: int = 0):
        self.active = False
        self.failures = failures
        self.starts = 0
        self.closed = False

    def start(self) -> None:
        self.starts += 1
        if self.starts <= self.failures:
            raise RuntimeError("Error starting stream: Unanticipated host error")
        self.active = True

    def abort(self) -> None:
        self.active = False

    def close(self) -> None:
        self.closed = True
        self.active = False


def make_player(stream: FakeStream) -> PCMPlayer:
    p = PCMPlayer.__new__(PCMPlayer)
    p._stream = stream
    p._stream_sample_rate = 48000
    p._stream_channels = 2
    p._stream_sd_dtype = "float32"
    p._lock = threading.RLock()
    p._selected_device_id = ""
    p._replacing_stream = False
    p._callback_carry = None
    return p


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(player, "audio_log", logging.getLogger("test.audio"))
    monkeypatch.setattr(player.time, "sleep", lambda _s: None)
    # Real device queries and a PortAudio re-init have no place in a
    # unit test; individual tests opt back in to what they need.
    monkeypatch.setattr(
        PCMPlayer, "_reinit_portaudio", staticmethod(lambda: None)
    )
    monkeypatch.setattr(
        player, "resolve_fallback_output_device", lambda *a, **k: (None, None)
    )


def test_starts_without_reopening_when_the_first_attempt_works():
    stream = FakeStream()
    p = make_player(stream)
    p._open_output_stream = lambda *a, **k: pytest.fail("should not reopen")

    p._ensure_stream_started()

    assert stream.active
    assert not stream.closed


def test_skips_start_when_the_stream_is_already_active():
    stream = FakeStream()
    stream.active = True
    p = make_player(stream)
    p._open_output_stream = lambda *a, **k: pytest.fail("should not reopen")

    p._ensure_stream_started()

    assert stream.starts == 0


def test_reopens_once_the_driver_releases_the_device():
    dead = FakeStream(failures=1)
    fresh = FakeStream()
    p = make_player(dead)

    def reopen(*_args, **_kwargs):
        p._stream = fresh

    p._open_output_stream = reopen

    p._ensure_stream_started()

    assert dead.closed
    assert fresh.active
    assert p._stream is fresh


def test_raises_the_last_failure_when_every_attempt_fails():
    p = make_player(FakeStream(failures=1))

    def reopen(*_args, **_kwargs):
        p._stream = FakeStream(failures=1)

    p._open_output_stream = reopen

    with pytest.raises(RuntimeError, match="Unanticipated host error"):
        p._ensure_stream_started()


def test_routes_to_another_hostapi_copy_when_the_preferred_copy_refuses(
    monkeypatch,
):
    """A Spatial Sound change leaves the WASAPI copy unstartable while
    the MME copy of the same speakers still works. The retry must walk
    to that copy instead of failing the play."""
    p = make_player(FakeStream(failures=99))
    monkeypatch.setattr(
        player, "resolve_fallback_output_device", lambda *a, **k: (77, "mme")
    )
    fresh = FakeStream()
    opened: list = []

    def reopen(_rate, _channels, _dtype, device_override=None):
        opened.append(device_override)
        p._stream = (
            fresh if device_override == 77 else FakeStream(failures=99)
        )

    p._open_output_stream = reopen

    p._ensure_stream_started()

    assert 77 in opened
    assert p._stream is fresh
    assert fresh.active


def test_falls_back_to_the_system_default_without_dropping_the_selection(
    monkeypatch,
):
    p = make_player(FakeStream(failures=99))
    p._selected_device_id = "Speakers (Realtek(R) Audio)"
    seen: list[str] = []

    def reopen(_rate, _channels, _dtype, device_override=None):
        seen.append(p._selected_device_id)
        p._stream = (
            FakeStream()
            if p._selected_device_id == ""
            else FakeStream(failures=99)
        )

    p._open_output_stream = reopen

    p._ensure_stream_started()

    assert seen[-1] == "", "the default tier must clear the selection"
    assert p._selected_device_id == "Speakers (Realtek(R) Audio)", (
        "the user's device choice must survive a temporary fallback"
    )
    assert p._stream is not None and p._stream.active


def test_drops_a_stream_that_never_started(monkeypatch):
    """Reusing the corpse is what spread one bad endpoint across the
    whole queue: every following track took the kept-stream-open path
    and failed its start again."""
    dead = FakeStream(failures=99)
    p = make_player(dead)

    def reopen(*_args, **_kwargs):
        p._stream = FakeStream(failures=99)

    p._open_output_stream = reopen

    with pytest.raises(RuntimeError):
        p._ensure_stream_started()

    assert p._stream is None
    assert dead.closed

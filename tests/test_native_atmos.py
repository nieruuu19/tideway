"""Native Atmos routing + test panel.

The native pipeline itself shells out to PowerShell, so these tests
pin the *routing* decisions with a fake helper: Atmos tracks go to the
native engine, everything else stays on PyAV, and a native failure
falls back to the PCM path instead of failing the play. A streamed
Atmos track has one extra step — its DASH segments are joined into a
local file first — which is exercised against a fake HTTP session.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from app.audio import player as player_mod
from app.audio.player import PCMPlayer, StreamInfo


def _atmos_info(
    track_gain: float | None = None,
    track_peak: float | None = None,
) -> StreamInfo:
    return StreamInfo(
        source="local",
        codec="eac3",
        sample_rate_hz=48000,
        audio_mode="DOLBY_ATMOS",
        track_replay_gain_db=track_gain,
        track_peak=track_peak,
    )


class _FakeNative:
    """Stand-in for NativeAtmosPlayer. Records commands; `state()`
    returns a paused reply so the position monitor stays quiet."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.closed = False
        self.fail_load = False
        self.gain_db = 0.0
        self.gain_applied_db = 0.0
        # Active spatial format of the endpoint, as the helper reports
        # it. "" = Spatial Sound off.
        self.spatial = ""
        self._pos = 0
        self._lock = threading.Lock()

    def load(
        self, path: str, autoplay: bool = True, adaptive: bool = False
    ) -> dict:
        self.calls.append(("load", (path, autoplay, adaptive)))
        if self.fail_load:
            raise RuntimeError("helper exploded")
        return {
            "ok": True,
            "pos_ms": 0,
            "dur_ms": 10000,
            "state": "Paused",
            "decoders": ["DolbyDDPDecMft.dll"],
            "device": "Speakers (Realtek(R) Audio)",
            "spatial_supported": True,
            "spatial_format": self.spatial,
        }

    def play(self) -> dict:
        self.calls.append(("play", None))
        return {}

    def pause(self) -> dict:
        self.calls.append(("pause", None))
        return {}

    def seek_ms(self, ms: int) -> dict:
        self.calls.append(("seek", ms))
        with self._lock:
            self._pos = ms
        return {}

    def set_volume(self, percent: float) -> dict:
        self.calls.append(("volume", percent))
        return {}

    def set_muted(self, muted: bool) -> dict:
        self.calls.append(("mute", muted))
        return {}

    def set_gain_db(self, gain_db: float) -> dict:
        self.calls.append(("gain", gain_db))
        self.gain_db = gain_db
        # MediaPlayer's volume stage caps at unity, so anything above
        # the user's headroom lands quieter than requested.
        applied = min(gain_db, 0.0)
        self.gain_applied_db = applied
        return {"ok": True, "gain_db": gain_db, "gain_applied_db": applied}

    def state(self) -> dict:
        with self._lock:
            pos = self._pos
        return {
            "ok": True,
            "pos_ms": pos,
            "dur_ms": 10000,
            "state": "Paused",
            "decoders": ["DolbyDDPDecMft.dll"],
            "device": "Speakers",
            "spatial_supported": True,
            "spatial_format": self.spatial,
        }

    def close(self) -> None:
        self.closed = True

    def stop(self) -> dict:
        self.calls.append(("stop", None))
        return {}

    @property
    def running(self) -> bool:
        return not self.closed


@pytest.fixture()
def local_atmos(tmp_path, monkeypatch):
    path = tmp_path / "Track [Dolby Atmos].m4a"
    path.write_bytes(b"\x00" * 64)
    monkeypatch.setattr(player_mod, "_local_file_is_atmos", lambda p: True)
    monkeypatch.setattr(
        player_mod,
        "_probe_local_stream_info",
        lambda p: (_atmos_info(), 10.0),
    )
    fake = _FakeNative()
    monkeypatch.setattr(player_mod, "NativeAtmosPlayer", lambda: fake)
    monkeypatch.setattr(player_mod, "native_atmos", _StubSupport(True))
    return str(path), fake


class _StubSupport:
    def __init__(self, value: bool) -> None:
        self._value = value

    def supported(self) -> bool:
        return self._value


def _record_and_fail(called: list[str]):
    def _resolve(*_args, **_kwargs):
        called.append("pcm")
        raise RuntimeError("stop here")

    return _resolve


def _player(path: str, enabled: bool = True) -> PCMPlayer:
    return PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: path if tid == "track" else None,
        native_atmos_enabled=lambda: enabled,
    )


def test_native_play_track_plays_and_reports_position(local_atmos):
    path, fake = local_atmos
    p = _player(path)

    snap = p.play_track("track")

    assert snap.state == "playing"
    assert snap.track_id == "track"
    assert snap.duration_ms == 10000
    assert snap.stream_info is not None
    assert snap.stream_info.audio_mode == "DOLBY_ATMOS"
    assert ("load", (path, False, False)) in fake.calls
    assert ("volume", 100) in fake.calls
    assert ("play", None) in fake.calls
    assert p.native_atmos_status() is not None
    assert p.native_atmos_status()["decoders"] == ["DolbyDDPDecMft.dll"]

    p.pause()
    assert ("pause", None) in fake.calls

    p.seek(0.5)
    assert ("seek", 5000) in fake.calls
    assert p.snapshot().position_ms == 5000

    p.stop()
    assert fake.closed is True
    assert p.native_atmos_status() is None
    assert p.snapshot().state == "idle"


def test_native_load_stays_silent_until_play(local_atmos):
    """Launch restores the persisted track through load() and expects
    it paused; starting audio here is what made Tideway play by itself
    on open."""
    path, fake = local_atmos
    p = _player(path)

    snap = p.load("track")

    assert snap.state == "paused"
    assert ("play", None) not in fake.calls
    assert p.native_atmos_status() is not None

    assert p.play().state == "playing"
    assert ("play", None) in fake.calls


def test_native_disabled_keeps_pcm_path(local_atmos, monkeypatch):
    path, fake = local_atmos
    p = _player(path, enabled=False)
    called: list[str] = []
    monkeypatch.setattr(p, "_resolve_source", _record_and_fail(called))

    snap = p.load("track")

    assert called == ["pcm"]
    assert snap.state == "error"
    assert fake.calls == []


def test_native_failure_falls_back_to_pcm(local_atmos, monkeypatch):
    path, fake = local_atmos
    fake.fail_load = True
    p = _player(path)
    called: list[str] = []
    monkeypatch.setattr(p, "_resolve_source", _record_and_fail(called))

    snap = p.load("track")

    assert called == ["pcm"]
    assert snap.state == "error"
    assert p.native_atmos_status() is None
    assert fake.closed is True


def test_non_atmos_local_file_is_not_native(local_atmos, monkeypatch):
    path, fake = local_atmos
    monkeypatch.setattr(player_mod, "_local_file_is_atmos", lambda p: False)
    p = _player(path)
    called: list[str] = []
    monkeypatch.setattr(p, "_resolve_source", _record_and_fail(called))

    p.load("track")

    assert called == ["pcm"]
    assert fake.calls == []


def test_preload_skips_atmos_tracks(local_atmos):
    path, fake = local_atmos
    p = _player(path)

    result = p.preload("track")

    assert result["native"] is True
    assert p._preload is None
    assert fake.calls == []


def test_native_applies_tag_gain_before_playing(local_atmos, monkeypatch):
    path, fake = local_atmos
    monkeypatch.setattr(
        player_mod,
        "_probe_local_stream_info",
        lambda p: (_atmos_info(track_gain=-7.5, track_peak=0.8), 10.0),
    )
    p = _player(path)
    p.set_replaygain("track", 0.0, True)

    p.play_track("track")

    assert ("gain", -7.5) in fake.calls
    names = [call[0] for call in fake.calls]
    # Ahead of play(): a level drop on the first sample is audible.
    assert names.index("gain") < names.index("play")
    status = p.native_atmos_status()
    assert status["gain_db"] == -7.5
    assert status["gain_source"] == "tags"
    assert p.replaygain_state()["applied_db"] == -7.5
    assert p.replaygain_state()["source"] == "tags"


def test_native_uses_cached_loudness_when_tags_missing(local_atmos, monkeypatch):
    path, fake = local_atmos
    monkeypatch.setattr(
        player_mod.loudness, "cached", lambda p: {"lufs": -12.0, "peak": 0.95}
    )
    p = _player(path)
    p.set_replaygain("track", 0.0, True)

    p.load("track")

    # -16 LUFS target against a -12 LUFS measure.
    assert ("gain", -4.0) in fake.calls
    status = p.native_atmos_status()
    assert status["gain_source"] == "measured"
    assert status["loudness_lufs"] == -12.0


def test_native_scans_missing_loudness_and_pushes_result(local_atmos, monkeypatch):
    path, fake = local_atmos
    monkeypatch.setattr(player_mod.loudness, "cached", lambda p: None)
    monkeypatch.setattr(
        player_mod.loudness, "analyze", lambda p: {"lufs": -20.0, "peak": 0.5}
    )
    p = _player(path)
    p.set_replaygain("track", 0.0, True)

    snap = p.play_track("track")

    assert snap.state == "playing"
    # Unmeasured at load: the track starts flat, then levels up.
    assert ("gain", 0.0) in fake.calls
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if p.native_atmos_status()["gain_source"] == "measured":
            break
        time.sleep(0.02)
    status = p.native_atmos_status()
    assert status["gain_source"] == "measured"
    # -16 target against a -20 LUFS measure needs +4 dB, but unity
    # volume has no headroom for it.
    assert status["gain_db"] == pytest.approx(4.0)
    assert status["gain_applied_db"] == 0.0


def test_mode_change_repushes_native_gain(local_atmos, monkeypatch):
    path, fake = local_atmos
    monkeypatch.setattr(
        player_mod,
        "_probe_local_stream_info",
        lambda p: (_atmos_info(track_gain=-6.0, track_peak=0.9), 10.0),
    )
    p = _player(path)
    p.load("track")
    fake.calls.clear()

    p.set_replaygain("track", 2.0, True)

    assert ("gain", -4.0) in fake.calls


def test_replaygain_off_leaves_native_level_alone(local_atmos):
    path, fake = local_atmos
    p = _player(path)

    p.load("track")

    assert ("gain", 0.0) in fake.calls
    assert p.native_atmos_status()["gain_source"] == "off"


def test_native_badge_tracks_the_endpoint_spatial_format(local_atmos):
    """The Atmos badge answers "are the objects being placed right
    now?", so it follows the endpoint's active spatial format: Dolby
    Atmos while the Atmos renderer is on, Dolby Audio as soon as Windows
    switches to Sonic (or spatial goes off) mid-track."""
    path, fake = local_atmos
    fake.spatial = "atmos-headphones"
    p = _player(path)

    snap = p.play_track("track")

    assert snap.immersive == "atmos"
    assert p.native_atmos_status()["spatial_format"] == "atmos-headphones"

    fake.spatial = "windows-sonic"
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if p.snapshot().immersive == "dolby_audio":
            break
        time.sleep(0.02)

    assert p.snapshot().immersive == "dolby_audio"

    # Dropping the native pipeline drops the classification with it.
    p.stop()
    assert p.snapshot().immersive is None


def test_native_monitor_resumes_a_mid_track_dropout(local_atmos):
    """A Spatial Sound change can park MediaPlayer at Paused mid-track.
    Leaving our state at "playing" would be silent playback that never
    advances; the monitor resumes it (rate limited), and only the real
    end-of-track is reported as `ended`."""
    path, fake = local_atmos
    p = _player(path)

    p.play_track("track")
    plays_after_load = len([c for c in fake.calls if c[0] == "play"])

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if len([c for c in fake.calls if c[0] == "play"]) > plays_after_load:
            break
        time.sleep(0.02)

    assert len([c for c in fake.calls if c[0] == "play"]) > plays_after_load
    assert p.snapshot().state == "playing"
    # 5 s into a 10 s track is a drop-out, not EOF.
    assert p.native_atmos_status()["state"] == "Paused"


def test_native_helper_survives_an_atmos_track_change(tmp_path, monkeypatch):
    """Atmos -> Atmos keeps the PowerShell host: the next track only
    needs `Source` repointed, so paying the measured ~0.5 s spawn
    again on every skip is pure loss."""
    first = tmp_path / "One [Dolby Atmos].m4a"
    second = tmp_path / "Two [Dolby Atmos].m4a"
    for path in (first, second):
        path.write_bytes(b"\x00" * 64)
    monkeypatch.setattr(player_mod, "_local_file_is_atmos", lambda p: True)
    monkeypatch.setattr(
        player_mod,
        "_probe_local_stream_info",
        lambda p: (_atmos_info(), 10.0),
    )
    monkeypatch.setattr(player_mod, "native_atmos", _StubSupport(True))
    made: list[_FakeNative] = []

    def _make() -> _FakeNative:
        helper = _FakeNative()
        made.append(helper)
        return helper

    monkeypatch.setattr(player_mod, "NativeAtmosPlayer", _make)
    paths = {"a": str(first), "b": str(second)}
    p = PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: paths.get(tid),
        native_atmos_enabled=lambda: True,
    )

    p.play_track("a")
    calls_after_first = len(made[0].calls)

    p.load("b")

    assert len(made) == 1
    # The outgoing track is silenced before the new source is opened.
    assert made[0].calls[calls_after_first] == ("pause", None)
    loads = [c for c in made[0].calls if c[0] == "load"]
    assert [c[1][0] for c in loads] == [str(first), str(second)]
    assert made[0].closed is False
    assert p.native_atmos_status() is not None

    # A full stop still disposes of it.
    p.stop()
    assert made[0].closed is True


def test_pcm_load_right_after_an_atmos_track_skips_the_preload(
    local_atmos, monkeypatch
):
    """A preload held during native playback must not be adopted: the
    native engine has no sounddevice stream, so adopting fed a queue
    nothing drained and the helper kept playing the old track under a
    silently "playing" new one."""
    path, fake = local_atmos
    p = _player(path)
    p.load("track")

    class _Staged:
        def __init__(self) -> None:
            self.track_id = "next"
            self.quality = None
            self.stop_flag = threading.Event()
            self.decoder = None
            self.thread = threading.Thread()

    p._preload = _Staged()

    def _unexpected(pre):
        raise AssertionError("adopted into a player with no output stream")

    monkeypatch.setattr(p, "_adopt_preload_locked", _unexpected)
    monkeypatch.setattr(p, "_resolve_source", _record_and_fail(["pcm"]))

    snap = p.load("next")

    assert snap.state == "error"
    # Falling through to the slow path silences the old helper and
    # parks the process for the next Atmos track, instead of making an
    # Atmos -> PCM -> Atmos queue pay the spawn twice.
    assert p.native_atmos_status() is None
    assert fake.closed is False
    assert ("pause", None) in fake.calls
    assert ("mute", True) in fake.calls
    assert ("volume", 0.0) in fake.calls


def test_the_parked_helper_serves_the_next_atmos_track(
    local_atmos, monkeypatch
):
    """Atmos -> PCM -> Atmos reuses one helper process.

    The PCM leg parks the helper; the next Atmos click opens its
    source in that same process, which is where the 0.4-0.8 s spawn
    would otherwise land.
    """
    path, fake = local_atmos
    p = _player(path)
    p.load("track")
    monkeypatch.setattr(p, "_resolve_source", _record_and_fail(["pcm"]))

    p.load("next")
    assert fake.closed is False

    p.play_track("track")

    assert fake.closed is False
    loads = [c for c in fake.calls if c[0] == "load"]
    assert [c[1][0] for c in loads] == [path, path]
    # Opened paused and silent, and the click starts it as usual.
    assert p.snapshot().state == "playing"
    assert ("play", None) in fake.calls


def test_a_full_stop_disposes_of_a_parked_helper(local_atmos, monkeypatch):
    path, fake = local_atmos
    p = _player(path)
    p.load("track")
    monkeypatch.setattr(p, "_resolve_source", _record_and_fail(["pcm"]))
    p.load("next")
    assert fake.closed is False

    p.stop()

    assert fake.closed is True


# ---------------------------------------------------------------------------
# Streamed Atmos: DASH segments -> one local file -> native pipeline
# ---------------------------------------------------------------------------

URLS = [
    "https://cdn.example/init.mp4",
    "https://cdn.example/seg1.m4s",
    "https://cdn.example/seg2.m4s",
    "https://cdn.example/seg3.m4s",
]
BODIES = {
    URLS[0]: b"INIT",
    URLS[1]: b"seg1",
    URLS[2]: b"seg2",
    URLS[3]: b"seg3",
}


def _atmos_stream_info(track_gain: float | None = None) -> StreamInfo:
    return StreamInfo(
        source="stream",
        codec="eac3",
        sample_rate_hz=48000,
        audio_quality="DOLBY_ATMOS",
        audio_mode="DOLBY_ATMOS",
        track_replay_gain_db=track_gain,
    )


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 65536):
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self._bodies = bodies
        self.requests: list[str] = []

    def get(self, url: str, **_kwargs) -> _FakeResponse:
        self.requests.append(url)
        return _FakeResponse(self._bodies.get(url, b""))


class _GatedSession(_FakeSession):
    """Session whose requests park until `gate` opens, so a test can
    look at a buffer that is still in flight."""

    def __init__(self, bodies: dict[str, bytes], gate: threading.Event) -> None:
        super().__init__(bodies)
        self.gate = gate
        self.grabbed = threading.Event()

    def get(self, url: str, **kwargs) -> _FakeResponse:
        self.grabbed.set()
        self.gate.wait(timeout=10)
        return super().get(url, **kwargs)


@pytest.fixture()
def stream_player(tmp_path, monkeypatch):
    fake = _FakeNative()
    fake.spatial = "atmos-headphones"
    monkeypatch.setattr(player_mod, "NativeAtmosPlayer", lambda: fake)
    monkeypatch.setattr(player_mod, "native_atmos", _StubSupport(True))
    p = PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: None,
        native_atmos_enabled=lambda: True,
    )
    p._native_stream_dir = str(tmp_path / "streams")
    return p, fake


def _resolves_atmos(urls=None, info=None, duration=123.0):
    def _resolve(*_args, **_kwargs):
        return (
            list(urls or URLS),
            duration,
            info or _atmos_stream_info(),
            {},
            None,
        )

    return _resolve


def test_streamed_atmos_is_buffered_and_played_native(
    stream_player, monkeypatch
):
    """The download path a TIDAL Atmos play actually takes: segments
    joined in order into one fMP4, opened by the native helper, and
    classified as Atmos while the endpoint's Atmos renderer is on."""
    p, fake = stream_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    assert snap.duration_ms == 123000
    assert snap.stream_info is not None
    assert snap.stream_info.audio_mode == "DOLBY_ATMOS"
    assert snap.immersive == "atmos"
    loads = [call for call in fake.calls if call[0] == "load"]
    assert len(loads) == 1
    buffered = loads[0][1][0]
    assert buffered.endswith("123.m4a")
    assert Path(buffered).read_bytes() == b"INITseg1seg2seg3"
    assert session.requests == URLS
    # The asset's own normalization tags ride the same gain channel as
    # a downloaded file's, so no R128 scan of the temp file is needed.
    assert ("gain", -7.5) in fake.calls
    assert p.native_atmos_status()["gain_source"] == "tags"


def test_a_streamed_atmos_restore_does_not_start_audio(
    stream_player, monkeypatch
):
    """The persisted track on launch is usually an Atmos stream: the
    restore load() buffers and opens it, but must not make a sound."""
    p, fake = stream_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)

    snap = p.load("123", quality="atmos")

    assert snap.state == "paused"
    assert ("play", None) not in fake.calls
    assert p._source_path is not None


def test_streamed_atmos_reuses_the_buffered_file(stream_player, monkeypatch):
    p, fake = stream_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())

    p.play_track("123", quality="atmos")
    first = [c for c in fake.calls if c[0] == "load"][0][1][0]
    p.stop()
    session.requests.clear()

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    assert session.requests == []
    assert [c for c in fake.calls if c[0] == "load"][-1][1][0] == first


def test_stream_buffer_reuses_prefetched_segments(stream_player, monkeypatch):
    p, _fake = stream_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)

    path = p._buffer_atmos_segments(
        "123", list(URLS), {0: b"INIT", 1: b"seg1"}, p._load_seq
    )

    assert path is not None
    assert Path(path).read_bytes() == b"INITseg1seg2seg3"
    assert session.requests == [URLS[2], URLS[3]]


def test_stream_buffer_abandons_on_a_newer_load(stream_player, monkeypatch):
    """A click landing while an Atmos buffer is running bumps the load
    sequence; the buffered download must stop and leave no partial file
    for the native pipeline to trip over."""
    p, _fake = stream_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    p._load_seq = 5

    path = p._buffer_atmos_segments("123", list(URLS), {}, load_seq=4)

    assert path is None
    streamed = Path(p._native_stream_dir)
    assert not (streamed / "123.m4a").exists()
    assert list(streamed.glob("*.part")) == []


def test_non_atmos_asset_stays_on_the_pcm_engine(stream_player, monkeypatch):
    """Requesting Atmos for a release without an Atmos asset resolves to
    the lossless fallback: nothing is downloaded, the native helper is
    never opened, and the fallback answer is cached under the "atmos"
    key so the PCM path behind it doesn't repeat the catalog walk."""
    p, fake = stream_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)
    fallback = StreamInfo(
        source="stream", codec="flac", audio_quality="LOSSLESS"
    )
    calls: list[str] = []

    def _resolve(*_args, **_kwargs):
        calls.append("resolve")
        if len(calls) == 1:
            return list(URLS), 123.0, fallback, {}, None
        raise RuntimeError("stop here")

    monkeypatch.setattr(p, "_resolve_source", _resolve)

    snap = p.load("123", quality="atmos")

    assert snap.state == "error"
    assert fake.calls == []
    assert session.requests == []
    cached = p._manifest_cache.lookup(("123", "atmos"))
    assert cached is not None
    assert cached[2] is fallback


def test_preload_publishes_streamed_atmos_as_dash(dash_player, monkeypatch):
    """Preloading the next Atmos track starts the very stream the click
    will open, so the click skips the network instead of joining a
    second, competing download — the parity with Apple Music / Spotify's
    look-ahead buffering. The PCM preload slot stays empty: the whole
    asset goes to the native pipeline, not to a PyAV decode of the bed."""
    p, _fake = dash_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())

    result = p.preload("123", "atmos")

    assert result["native"] is True
    assert result["started"] is True
    assert p._preload is None
    published = p._dash_preload_for("123")
    assert published is not None
    assert published.done.wait(timeout=5.0)
    buffered = p._native_stream_hit("123")
    assert buffered is not None
    assert Path(buffered.path).read_bytes() == b"INITseg1seg2seg3"
    assert session.requests == URLS


def test_load_uses_the_preloaded_file_without_new_requests(
    dash_player, monkeypatch
):
    p, fake = dash_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())
    p.preload("123", "atmos")
    assert p._dash_preload_for("123").done.wait(timeout=5.0)
    session.requests.clear()

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    assert session.requests == []
    assert [c for c in fake.calls if c[0] == "load"]


def test_load_adopts_an_in_flight_atmos_preload(dash_player, monkeypatch):
    """A click landing while the look-ahead is still downloading takes
    that session over instead of starting a second one for the same
    segments (two concurrent whole-asset pulls is exactly the burst
    TIDAL's gate treats as abuse), and hands the helper its manifest
    rather than waiting for the joined file."""
    p, fake = dash_player
    gate = threading.Event()
    session_stub = _GatedSession(BODIES, gate)
    monkeypatch.setattr("app.http.SESSION", session_stub)
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())

    p.preload("123", "atmos")
    published = p._dash_preload_for("123")
    assert published is not None
    assert session_stub.grabbed.wait(timeout=5.0)
    threading.Timer(0.3, gate.set).start()

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    loads = [c for c in fake.calls if c[0] == "load"]
    assert loads[-1][1][0] == published.url
    assert loads[-1][1][2] is True
    assert p._dash_atmos is published
    assert p._dash_preload_for("123") is None

    assert published.done.wait(timeout=5.0)
    # One download of each segment: the preload's. The load adopted it.
    assert session_stub.requests.count(URLS[1]) == 1


def test_preload_of_another_track_retires_the_first(dash_player, monkeypatch):
    p, _fake = dash_player
    gate = threading.Event()
    session_stub = _GatedSession(BODIES, gate)
    monkeypatch.setattr("app.http.SESSION", session_stub)
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())

    p.preload("123", "atmos")
    first = p._dash_preload_for("123")
    assert first is not None
    assert session_stub.grabbed.wait(timeout=5.0)
    p.preload("456", "atmos")
    gate.set()

    assert first.cancel.is_set()
    assert p._dash_preload_for("123") is None
    second = p._dash_preload_for("456")
    assert second is not None
    assert second.done.wait(timeout=5.0)


def test_stop_cancels_an_in_flight_atmos_preload(dash_player, monkeypatch):
    p, _fake = dash_player
    gate = threading.Event()
    session_stub = _GatedSession(BODIES, gate)
    monkeypatch.setattr("app.http.SESSION", session_stub)
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())

    p.preload("123", "atmos")
    session = p._dash_preload_for("123")
    assert session is not None
    assert session_stub.grabbed.wait(timeout=5.0)
    p.stop()
    gate.set()

    assert session.cancel.is_set()
    assert session.done.wait(timeout=5.0)
    assert p._dash_preload_for("123") is None
    assert p._native_stream_hit("123") is None


def test_preload_without_an_atmos_asset_downloads_nothing(
    dash_player, monkeypatch
):
    p, _fake = dash_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)
    fallback = StreamInfo(
        source="stream", codec="flac", audio_quality="LOSSLESS"
    )
    monkeypatch.setattr(
        p,
        "_resolve_source",
        lambda *_a, **_k: (list(URLS), 123.0, fallback, {}, None),
    )

    result = p.preload("123", "atmos")

    assert result["native"] is False
    assert p._dash_preload_for("123") is None
    assert session.requests == []
    # The fallback is remembered for the PCM play that follows, without
    # walking the Atmos lookup a second time.
    assert p._manifest_cache.lookup(("123", "atmos"))[2] is fallback


def test_preload_with_streaming_off_downloads_nothing(
    stream_player, monkeypatch
):
    """With DASH turned off there is no session to pre-stage, and a
    whole-file download would only be repeated when the click lands —
    the play path buffers it then."""
    p, _fake = stream_player
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)

    def _boom(*_args, **_kwargs):
        raise AssertionError("nothing to resolve without streaming")

    monkeypatch.setattr(p, "_resolve_source", _boom)

    result = p.preload("123", "atmos")

    assert result["native"] is True
    assert result.get("started") is None
    assert session.requests == []


def test_lossless_quality_never_buffers_a_stream(stream_player, monkeypatch):
    p, _fake = stream_player

    def _boom(*_args, **_kwargs):
        raise AssertionError("only the atmos tier buffers a stream")

    monkeypatch.setattr(p, "_resolve_source", _boom)

    assert (
        p._native_candidate("123", "hi_res_lossless", allow_stream=True)
        is None
    )


# ---------------------------------------------------------------------------
# Buffered Atmos files outlive the process
# ---------------------------------------------------------------------------


def _quiet_player(cache_dir) -> tuple[PCMPlayer, _FakeNative]:
    fake = _FakeNative()
    p = PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: None,
        native_atmos_enabled=lambda: True,
    )
    p._native_stream_dir = str(cache_dir)
    return p, fake


def test_buffered_atmos_survives_a_restart(tmp_path, monkeypatch):
    """The whole point of the persistent cache: the second launch opens
    the file the first one downloaded, with TIDAL's normalization tags
    restored from the sidecar (no re-resolve, no re-download, no R128
    scan of every track)."""
    cache_dir = tmp_path / "cache"
    first, fake = _quiet_player(cache_dir)
    monkeypatch.setattr(player_mod, "NativeAtmosPlayer", lambda: fake)
    monkeypatch.setattr(player_mod, "native_atmos", _StubSupport(True))
    session = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session)
    monkeypatch.setattr(
        first, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    first.set_replaygain("track", 0.0, True)

    assert first.play_track("123", quality="atmos").state == "playing"
    assert session.requests == URLS
    assert (cache_dir / "123.m4a.json").exists()

    # A fresh player with an empty in-memory cache is what a restart
    # looks like from the cache's point of view.
    session.requests.clear()
    fake.calls.clear()
    second, _fake = _quiet_player(cache_dir)
    monkeypatch.setattr(
        second,
        "_resolve_source",
        _record_and_fail(["resolved instead of using the cache"]),
    )
    second.set_replaygain("track", 0.0, True)

    snap = second.play_track("123", quality="atmos")

    assert snap.state == "playing"
    assert snap.duration_ms == 123000
    assert session.requests == []
    assert snap.stream_info is not None
    assert snap.stream_info.source == "stream"
    assert snap.stream_info.audio_quality == "DOLBY_ATMOS"
    assert ("gain", -7.5) in fake.calls


def test_expired_record_is_readopted_from_disk(stream_player, monkeypatch):
    """The TTL only retires the in-memory record; the file stays and is
    picked up again from its sidecar."""
    p, _fake = stream_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())
    p.load("123", quality="atmos")
    entry = p._native_stream_files["123"]
    entry.expires_at = 0.0

    again = p._native_stream_hit("123")

    assert again is not None
    assert again.path == entry.path
    assert again.info.audio_mode == "DOLBY_ATMOS"


def test_a_file_whose_sidecar_does_not_match_is_ignored(
    stream_player, monkeypatch
):
    p, _fake = stream_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(p, "_resolve_source", _resolves_atmos())
    snap = p.play_track("123", quality="atmos")
    assert snap.state == "playing"
    path = Path(p._native_stream_files["123"].path)
    # A truncated file (crash between publish and sidecar) must not be
    # handed to MediaPlayer on the strength of a stale sidecar.
    path.write_bytes(b"INIT")
    p._native_stream_files.clear()

    assert p._native_stream_hit("123") is None


def test_disk_budget_drops_the_oldest_buffered_file(stream_player, monkeypatch):
    p, _fake = stream_player
    directory = Path(p._native_stream_dir)
    directory.mkdir(parents=True, exist_ok=True)
    old = directory / "old.m4a"
    old.write_bytes(b"x" * 10)
    new = directory / "new.m4a"
    new.write_bytes(b"y" * 10)
    os.utime(old, (time.time() - 100, time.time() - 100))
    monkeypatch.setattr(player_mod, "_NATIVE_STREAM_DISK_MAX_BYTES", 10)

    p._prune_atmos_cache_files()

    assert not old.exists()
    assert new.exists()


def test_startup_sweep_removes_abandoned_buffers(stream_player):
    p, _fake = stream_player
    directory = Path(p._native_stream_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stale = directory / "123.m4a.99.part"
    stale.write_bytes(b"x")
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    fresh = directory / "456.m4a.99.part"
    fresh.write_bytes(b"x")
    orphan = directory / "789.m4a.json"
    orphan.write_text("{}")

    p._sweep_atmos_cache(str(directory))

    assert not stale.exists()
    assert fresh.exists()
    assert not orphan.exists()


# ---------------------------------------------------------------------------
# DASH streaming: play while the track downloads
# ---------------------------------------------------------------------------


class _StallingSession(_FakeSession):
    """Session that parks on one segment until `gate` opens, so the
    writer is still downloading while playback has already started."""

    def __init__(
        self, bodies: dict[str, bytes], stall_url: str, gate: threading.Event
    ) -> None:
        super().__init__(bodies)
        self.stall_url = stall_url
        self.gate = gate

    def get(self, url: str, **kwargs) -> _FakeResponse:
        if url == self.stall_url:
            self.gate.wait(timeout=10)
        return super().get(url, **kwargs)


@pytest.fixture()
def dash_player(tmp_path, monkeypatch):
    fake = _FakeNative()
    fake.spatial = "atmos-headphones"
    monkeypatch.setattr(player_mod, "NativeAtmosPlayer", lambda: fake)
    monkeypatch.setattr(player_mod, "native_atmos", _StubSupport(True))
    p = PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: None,
        native_atmos_enabled=lambda: True,
        native_atmos_dash=lambda: True,
    )
    p._native_stream_dir = str(tmp_path / "streams")
    return p, fake


def test_dash_atmos_opens_the_still_downloading_stream(
    dash_player, monkeypatch
):
    """The click no longer waits for 20-30 MB: as soon as the init
    segment and the first fragment exist the helper is handed a
    manifest, and the write that continues behind it lands in the cache
    as usual."""
    p, fake = dash_player
    gate = threading.Event()
    monkeypatch.setattr("app.http.SESSION", _StallingSession(BODIES, URLS[2], gate))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    session = p._dash_atmos
    assert session is not None
    path, autoplay, adaptive = [c for c in fake.calls if c[0] == "load"][0][1]
    assert path == session.url
    assert path.startswith("http://127.0.0.1:")
    assert path.endswith("/manifest.mpd")
    # An adaptive source, never autoplayed before its gain is set.
    assert (autoplay, adaptive) == (False, True)

    # Only part of the track has been written at this point.
    assert Path(session.directory, "init.mp4").read_bytes() == b"INIT"
    assert Path(session.directory, "seg-1.m4s").read_bytes() == b"seg1"
    assert not Path(session.directory, "seg-2.m4s").exists()
    assert not Path(session.cache_path).exists()
    assert player_mod.dash_server.is_published_directory(session.directory)

    gate.set()
    assert session.done.wait(timeout=5.0)
    cached = p._native_stream_hit("123")
    assert cached is not None
    assert Path(cached.path).read_bytes() == b"INITseg1seg2seg3"
    assert Path(f"{cached.path}.json").exists()


def test_a_dash_stream_that_cannot_prime_falls_back_to_the_file(
    dash_player, monkeypatch
):
    """A download that never reaches the first fragment must not hold
    the click: the stream is dropped and the track buffered the old
    way, which is what the PCM engine can follow."""
    p, fake = dash_player
    gate = threading.Event()
    monkeypatch.setattr(player_mod, "_DASH_PRIME_DEADLINE_S", 0.0)
    monkeypatch.setattr("app.http.SESSION", _StallingSession(BODIES, URLS[1], gate))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)
    threading.Timer(0.2, gate.set).start()

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    opened = [c for c in fake.calls if c[0] == "load"][0][1][0]
    assert opened.endswith("123.m4a")
    assert Path(opened).read_bytes() == b"INITseg1seg2seg3"
    assert p._dash_atmos is None


def test_dash_stream_refused_by_the_helper_falls_back_to_the_file(
    dash_player, monkeypatch
):
    """Media Foundation can reject a manifest it cannot make sense of.
    The download is still running, so the native path waits for the
    joined file and opens that rather than downloading the track again."""
    p, fake = dash_player
    gate = threading.Event()
    monkeypatch.setattr("app.http.SESSION", _StallingSession(BODIES, URLS[2], gate))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)
    real_load = fake.load

    def _load(path: str, autoplay: bool = True, adaptive: bool = False) -> dict:
        if adaptive:
            fake.calls.append(("load", (path, autoplay, adaptive)))
            raise RuntimeError("adaptive source rejected")
        return real_load(path, autoplay, adaptive)

    fake.load = _load
    # Let the writer finish shortly after the refusal.
    threading.Timer(0.2, gate.set).start()

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    loads = [c for c in fake.calls if c[0] == "load"]
    assert len(loads) == 2
    assert loads[0][1][2] is True
    assert loads[1][1][0].endswith("123.m4a")
    assert Path(loads[1][1][0]).read_bytes() == b"INITseg1seg2seg3"
    assert p._source_path == loads[1][1][0]
    assert p._dash_atmos is None


def test_a_second_atmos_track_retires_the_first_stream(
    dash_player, monkeypatch
):
    """Skipping from one DASH track to another must end the first
    download and stop serving its segments, while the new track's own
    stream survives the teardown that accompanies the switch."""
    p, _fake = dash_player
    gate = threading.Event()
    monkeypatch.setattr("app.http.SESSION", _StallingSession(BODIES, URLS[2], gate))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)

    assert p.play_track("123", quality="atmos").state == "playing"
    first = p._dash_atmos
    assert first is not None

    assert p.play_track("456", quality="atmos").state == "playing"
    second = p._dash_atmos

    assert second is not None
    assert second is not first
    assert first.cancel.is_set()
    assert not player_mod.dash_server.is_published_directory(first.directory)
    assert player_mod.dash_server.is_published_directory(second.directory)
    gate.set()


def test_stop_retires_the_dash_stream(dash_player, monkeypatch):
    p, _fake = dash_player
    gate = threading.Event()
    monkeypatch.setattr("app.http.SESSION", _StallingSession(BODIES, URLS[2], gate))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)
    assert p.play_track("123", quality="atmos").state == "playing"
    session = p._dash_atmos
    assert session is not None

    p.stop()

    assert session.cancel.is_set()
    assert p._dash_atmos is None
    assert not player_mod.dash_server.is_published_directory(session.directory)
    gate.set()


def test_startup_sweep_removes_abandoned_dash_directories(stream_player):
    """Segment directories are deleted when their session retires, but
    a crash (or a delete that lost a race with MediaPlayer) leaves one
    behind; the next launch takes it."""
    p, _fake = stream_player
    directory = Path(p._native_stream_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stale = directory / "dash-123-dead"
    stale.mkdir()
    (stale / "seg-1.m4s").write_bytes(b"x")
    old = time.time() - 7200
    os.utime(stale / "seg-1.m4s", (old, old))
    os.utime(stale, (old, old))
    fresh = directory / "dash-456-live"
    fresh.mkdir()
    (fresh / "seg-1.m4s").write_bytes(b"x")

    p._sweep_atmos_cache(str(directory))

    assert not stale.exists()
    assert fresh.exists()


def test_a_dash_manifest_is_not_scanned_for_loudness(stream_player):
    """The URL names a download in progress; an R128 pass over it would
    measure however many segments have landed. Leveling stays pending
    and the writer's completion starts the scan against the joined
    file."""
    p, _fake = stream_player
    p.set_replaygain("track", 0.0, True)

    gain, lufs, source = p._native_gain_plan(
        _atmos_stream_info(), "http://127.0.0.1:1234/dash/token/manifest.mpd"
    )

    assert (gain, lufs, source) == (0.0, None, "pending")


def test_a_seek_segment_is_served_from_the_download_it_already_made(
    dash_player, monkeypatch
):
    """A seek pulls the segment under the drop point out of order. When
    the sequential pass reaches it, the bytes on disk are joined — the
    track is downloaded once, and the joined file is still in order."""
    p, _fake = dash_player
    gate = threading.Event()
    session_stub = _StallingSession(BODIES, URLS[2], gate)
    monkeypatch.setattr("app.http.SESSION", session_stub)
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)

    p.load("123", quality="atmos")
    session = p._dash_atmos
    assert session is not None
    # The writer is parked on segment 2; a seek to near the end asks
    # the server for segment 3, which is three segments ahead of it.
    assert player_mod.dash_server.ensure_segment(session.publication, 3) is True
    assert Path(session.directory, "seg-3.m4s").read_bytes() == b"seg3"

    gate.set()
    assert session.done.wait(timeout=5.0)

    assert session_stub.requests.count(URLS[3]) == 1
    cached = p._native_stream_hit("123")
    assert cached is not None
    assert Path(cached.path).read_bytes() == b"INITseg1seg2seg3"




# ---------------------------------------------------------------------------
# Looking ahead into the native pipeline
# ---------------------------------------------------------------------------


class _NativeFactory:
    """`NativeAtmosPlayer` stand-in that hands out a new fake per call,
    so a test can tell the idle look-ahead helper apart from the one
    the click ends up using."""

    def __init__(self) -> None:
        self.made: list[_FakeNative] = []
        self._lock = threading.Lock()

    def __call__(self) -> _FakeNative:
        fake = _FakeNative()
        fake.spatial = "atmos-headphones"
        with self._lock:
            self.made.append(fake)
        return fake


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture()
def warm_player(tmp_path, monkeypatch):
    factory = _NativeFactory()
    monkeypatch.setattr(player_mod, "NativeAtmosPlayer", factory)
    monkeypatch.setattr(player_mod, "native_atmos", _StubSupport(True))
    p = PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: None,
        native_atmos_enabled=lambda: True,
        native_atmos_dash=lambda: True,
    )
    p._native_stream_dir = str(tmp_path / "streams")
    return p, factory


def test_preload_opens_the_next_track_in_an_idle_helper(
    warm_player, monkeypatch
):
    """The look-ahead is where the helper spawn and the first open get
    paid: the click is seconds later, and a source that is already open
    costs a volume change and a play."""
    p, factory = warm_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )

    result = p.preload("123", quality="atmos")

    assert result["started"] is True
    assert _wait_until(lambda: p._native_warm is not None)
    warm = p._native_warm
    session = p._dash_preload_for("123")
    assert session is not None
    assert warm in factory.made
    loads = [call for call in warm.calls if call[0] == "load"]
    assert loads == [("load", (session.url, False, True))]
    # Silent while it waits: opened, never heard.
    assert ("volume", 0.0) in warm.calls
    assert ("mute", True) in warm.calls
    assert ("play", None) not in warm.calls
    assert p._native is None


def test_the_click_adopts_the_warmed_helper(warm_player, monkeypatch):
    """A click on the preloaded track reuses that open source instead of
    spawning a helper and opening the manifest again."""
    p, factory = warm_player
    session_stub = _FakeSession(BODIES)
    monkeypatch.setattr("app.http.SESSION", session_stub)
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    p.set_replaygain("track", 0.0, True)

    assert p.preload("123", quality="atmos")["started"] is True
    assert _wait_until(lambda: p._native_warm is not None)
    warm = p._native_warm
    assert ("load", (p._dash_preload_for("123").url, False, True)) in warm.calls

    snap = p.play_track("123", quality="atmos")

    assert snap.state == "playing"
    # One helper, one open: the click only set the level and played.
    assert len(factory.made) == 1
    assert p._native is warm
    assert p._native_warm is None
    assert len([c for c in warm.calls if c[0] == "load"]) == 1
    assert ("play", None) in warm.calls
    assert ("volume", 100) in warm.calls
    assert ("mute", False) in warm.calls


def test_the_warmed_helper_is_dropped_with_its_session(
    warm_player, monkeypatch
):
    """The idle helper holds one stream. A look-ahead for another track
    retires that stream, so the helper goes with it rather than sitting
    on a URL the server no longer answers."""
    p, factory = warm_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )

    assert p.preload("123", quality="atmos")["started"] is True
    assert _wait_until(lambda: p._native_warm is not None)
    first = p._native_warm

    assert p.preload("456", quality="atmos")["started"] is True

    assert _wait_until(lambda: first.closed is True)
    assert _wait_until(
        lambda: p._native_warm is not None and p._native_warm is not first
    )


def test_stop_drops_the_warmed_helper(warm_player, monkeypatch):
    p, _factory = warm_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    assert p.preload("123", quality="atmos")["started"] is True
    assert _wait_until(lambda: p._native_warm is not None)
    warm = p._native_warm

    p.stop()

    assert p._native_warm is None
    assert warm.closed is True


# ---------------------------------------------------------------------------
# Cache resilience and rapid-skip coalescing
# ---------------------------------------------------------------------------


def test_a_deleted_stream_cache_dir_is_recreated(stream_player, monkeypatch):
    """Deleting the cache is a cache miss, not a broken track: the dir
    is recreated on the next use and the track downloads again."""
    import shutil

    p, fake = stream_player
    cache_dir = Path(p._atmos_stream_dir())
    assert cache_dir.is_dir()
    shutil.rmtree(cache_dir)
    assert not cache_dir.exists()

    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )

    snap = p.play_track("123", quality="atmos")

    assert cache_dir.is_dir()
    assert snap.state == "playing"
    assert (
        Path(p._atmos_cache_path("123")).read_bytes()
        == b"INITseg1seg2seg3"
    )


def test_a_native_load_overtaken_mid_open_is_never_played(
    stream_player, monkeypatch
):
    """A newer click landing while the helper opens the source must
    leave this track silent — the open helper is kept for the load
    that is already queued."""
    p, fake = stream_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    real_open = p._native_open

    def open_then_supersede(helper, path):
        reply = real_open(helper, path)
        with p._lock:
            p._load_seq += 1
            p._load_target_id = "456"
        return reply

    monkeypatch.setattr(p, "_native_open", open_then_supersede)

    snap = p.load("123", quality="atmos")

    loads = [call for call in fake.calls if call[0] == "load"]
    assert loads == [("load", (p._atmos_cache_path("123"), False, False))]
    assert ("play", None) not in fake.calls
    assert snap.state == "paused"
    # Kept warm: the newer load only has to swap Source.
    assert fake.closed is False
    assert p.native_atmos_status() is not None


def test_play_track_does_not_press_play_on_a_superseded_load(
    stream_player, monkeypatch
):
    p, fake = stream_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    real_open = p._native_open

    def open_then_supersede(helper, path):
        reply = real_open(helper, path)
        with p._lock:
            p._load_seq += 1
            p._load_target_id = "456"
        return reply

    monkeypatch.setattr(p, "_native_open", open_then_supersede)

    p.play_track("123", quality="atmos")

    assert ("play", None) not in fake.calls


def test_play_track_claims_the_click_before_waiting_for_the_lock(stream_player):
    """The click's sequence number has to be claimed on arrival, not
    once it reaches the pipeline lock. Rapid Next presses are separate
    HTTP calls; the ones queued behind an in-flight load can only mark
    it dead if their claim lands while they wait."""
    p, _fake = stream_player
    p._load_locked = lambda *_a, **_k: p.snapshot()
    p.play = lambda: p.snapshot()

    held = threading.Event()
    release = threading.Event()

    def hold_lock():
        with p._pipeline_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert held.wait(5.0)

    with p._lock:
        seq_before = p._load_seq

    click = threading.Thread(
        target=p.play_track, args=("123",), daemon=True
    )
    click.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with p._lock:
                if p._load_target_id == "123":
                    break
            time.sleep(0.01)
        with p._lock:
            assert p._load_target_id == "123"
            assert p._load_seq == seq_before + 1
            # Still queued: the claim did not have to wait for release.
            assert not release.is_set()
    finally:
        release.set()
        holder.join(5.0)
        click.join(5.0)
    assert not click.is_alive()


def test_a_queued_play_track_supersedes_the_inflight_one(stream_player, monkeypatch):
    """A burst of Next presses: the click already loading a track must
    stay silent once a newer click is waiting behind it, and only the
    newest one may start audio."""
    p, fake = stream_player
    monkeypatch.setattr("app.http.SESSION", _FakeSession(BODIES))
    monkeypatch.setattr(
        p, "_resolve_source", _resolves_atmos(info=_atmos_stream_info(-7.5))
    )
    real_open = p._native_open
    opened = threading.Event()
    release = threading.Event()
    opens: list[str] = []

    def open_then_wait(helper, path):
        opens.append(path)
        if len(opens) == 1:
            opened.set()
            assert release.wait(5.0)
        return real_open(helper, path)

    monkeypatch.setattr(p, "_native_open", open_then_wait)

    results: dict[str, object] = {}
    first = threading.Thread(
        target=lambda: results.setdefault(
            "first", p.play_track("123", quality="atmos")
        ),
        daemon=True,
    )
    first.start()
    assert opened.wait(5.0)
    # The user's next press, arriving while the first load is still
    # opening its source.
    second = threading.Thread(
        target=lambda: results.setdefault(
            "second", p.play_track("456", quality="atmos")
        ),
        daemon=True,
    )
    second.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with p._lock:
            if p._load_target_id == "456":
                break
        time.sleep(0.01)
    with p._lock:
        assert p._load_target_id == "456"

    release.set()
    first.join(5.0)
    second.join(5.0)
    assert not first.is_alive() and not second.is_alive()

    assert results["first"].state == "paused"
    assert results["second"].state == "playing"
    assert results["second"].track_id == "456"
    # One play, for the newest click only.
    assert fake.calls.count(("play", None)) == 1
    assert opens[-1] == p._atmos_cache_path("456")


def test_a_pcm_gapless_build_overtaken_mid_resolve_is_discarded(
    stream_player, monkeypatch
):
    """The PCM equivalent: a synthetic pipeline built for a track the
    user already skipped past is closed instead of swapped in, so no
    intermediate track is audible."""
    import types

    p, fake = stream_player
    p._stream = object()
    p._stream_sample_rate = 48000
    p._stream_sd_dtype = "float32"
    p._stream_channels = 2

    class _Decoder:
        def __init__(self):
            self.cancelled = False
            self.closed = False

        def cancel_source(self):
            self.cancelled = True

        def close(self):
            self.closed = True

    class _Thread:
        def is_alive(self):
            return False

    decoder = _Decoder()
    synthetic = types.SimpleNamespace(
        stop_flag=threading.Event(),
        decoder=decoder,
        thread=_Thread(),
        sample_rate=48000,
        sd_dtype="float32",
        channels=2,
        source_urls=None,
        source_path="/nonexistent/track.m4a",
    )

    def build(track_id, quality, match_rate=None):
        with p._lock:
            p._load_seq += 1
            p._load_target_id = "456"
        return synthetic

    monkeypatch.setattr(p, "_build_load_pipeline", build)

    p.load("123", quality="high")

    assert decoder.cancelled is True
    assert decoder.closed is True
    assert p._decoder is None  # never swapped in


def test_reloading_the_same_track_is_not_superseded(stream_player):
    p, _fake = stream_player
    with p._lock:
        p._load_seq = 1
        p._load_target_id = "track"
    assert p._load_superseded(1, "track") is False

    with p._lock:
        p._load_seq = 2
        p._load_target_id = "other"
    assert p._load_superseded(1, "track") is True

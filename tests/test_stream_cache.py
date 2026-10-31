"""Streamed-track disk cache (app/audio/stream_cache.py) + the
SegmentReader tee that fills it.

The cache exists so the second play of a streamed, non-Atmos track is
a local-file open instead of another download. These tests pin the
properties that make that safe: only complete segment sequences are
published, an abandoned download leaves nothing behind, and a resolve
hit never touches the network.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from app.audio import segment_reader as segment_reader_mod
from app.audio.player import PCMPlayer, StreamInfo
from app.audio.segment_reader import SegmentReader
from app.audio.stream_cache import StreamCache

URLS = [
    "https://cdn.example/init.mp4",
    "https://cdn.example/seg1.m4s",
    "https://cdn.example/seg2.m4s",
]
BODIES = {
    URLS[0]: b"INIT",
    URLS[1]: b"seg1",
    URLS[2]: b"seg2",
}
INFO = {
    "source": "stream",
    "codec": "flac",
    "bit_depth": 16,
    "sample_rate_hz": 44100,
    "audio_quality": "LOSSLESS",
    "audio_mode": None,
    "track_replay_gain_db": -6.0,
    "track_peak": None,
    "album_replay_gain_db": None,
    "album_peak": None,
}
META = {"title": "Song", "artist": "Artist", "album": "Album"}


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 65536):
        yield self._payload

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def get(self, url: str, **_kwargs) -> _FakeResponse:
        self.requests.append(url)
        return _FakeResponse(BODIES[url])

    def close(self) -> None:
        pass


@pytest.fixture()
def fake_http(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(segment_reader_mod.STREAM_SESSION, "get", session.get)
    return session


def test_segment_reader_tees_a_complete_track_into_the_cache(
    tmp_path, fake_http
):
    cache = StreamCache(str(tmp_path))
    sink = cache.sink("123", "high", len(URLS), 200.0, INFO, META)

    reader = SegmentReader(URLS, sink=sink)
    data = reader.read(-1)
    reader.close()

    assert data == b"INITseg1seg2"
    hit = cache.lookup("123", "high")
    assert hit is not None
    path, duration, info, meta = hit
    assert Path(path).read_bytes() == b"INITseg1seg2"
    assert duration == 200.0
    assert info["codec"] == "flac"
    assert meta == META
    # Only `.m4a` + sidecar remain: the `.part` was renamed.
    names = sorted(os.listdir(tmp_path))
    assert names == ["123-high.m4a", "123-high.m4a.json"]


def test_abandoned_download_leaves_no_file_or_part(tmp_path, fake_http):
    cache = StreamCache(str(tmp_path))
    sink = cache.sink("123", "high", len(URLS), 200.0, INFO, META)

    reader = SegmentReader(URLS, sink=sink)
    reader.read(4)  # init only; the rest never fetched
    reader.close()

    assert cache.lookup("123", "high") is None
    assert os.listdir(tmp_path) == []


def test_sink_rejects_an_out_of_order_segment(tmp_path):
    cache = StreamCache(str(tmp_path))
    sink = cache.sink("123", "high", 3, 200.0, INFO, META)

    sink.write(0, b"INIT")
    sink.write(2, b"seg2")  # gap: seg1 never arrived

    assert cache.lookup("123", "high") is None
    assert os.listdir(tmp_path) == []


def test_a_deleted_cache_dir_is_recreated(tmp_path):
    target = tmp_path / "stream_cache"
    cache = StreamCache(str(target))
    first = cache.path_for("123", "high")
    assert target.is_dir()

    for name in os.listdir(target):
        os.remove(target / name)
    os.rmdir(target)

    second = cache.path_for("123", "high")

    assert target.is_dir()
    assert first == second


def test_sweep_drops_expired_tracks(tmp_path):
    cache = StreamCache(str(tmp_path), max_age_s=0)
    sink = cache.sink("123", "high", 2, 200.0, INFO, META)
    sink.write(0, b"INIT")
    sink.write(1, b"seg1")
    assert cache.lookup("123", "high") is not None

    time.sleep(0.01)
    cache.sweep()

    assert cache.lookup("123", "high") is None
    assert os.listdir(tmp_path) == []


def test_disabled_cache_keeps_the_bytes_out_of_the_user_dir(
    tmp_path, monkeypatch
):
    # conftest points the real player's cache at a temp dir; this pins
    # that wiring so a suite run can never write into AppData.
    cache = StreamCache()
    assert Path(cache.directory()) == Path(
        os.environ["TIDEWAY_STREAM_CACHE_DIR"]
    )


def test_resolve_opens_a_finished_cache_file_without_the_network(tmp_path):
    cache = StreamCache(str(tmp_path))
    sink = cache.sink("123", "high", 2, 200.0, INFO, META)
    sink.write(0, b"INIT")
    sink.write(1, b"seg1")

    p = PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: None,
        native_atmos_enabled=lambda: False,
    )
    p._pcm_stream_cache = cache

    spec, duration, info, prefetched, meta = p._resolve_source(
        "123", "high"
    )

    assert isinstance(spec, str)
    assert Path(spec).read_bytes() == b"INITseg1"
    assert duration == 200.0
    assert isinstance(info, StreamInfo)
    assert info.codec == "flac"
    assert prefetched == {}
    assert meta == META


def test_atmos_requests_never_read_the_pcm_disk_cache(tmp_path):
    cache = StreamCache(str(tmp_path))
    sink = cache.sink("123", "atmos", 2, 200.0, INFO, META)
    sink.write(0, b"INIT")
    sink.write(1, b"seg1")

    p = PCMPlayer(
        lambda: None,
        local_lookup=lambda tid: None,
        native_atmos_enabled=lambda: False,
    )
    p._pcm_stream_cache = cache

    # The Atmos path resolves through TIDAL, not this cache; the
    # session getter returning None makes the miss explicit.
    with pytest.raises(Exception):
        p._resolve_source("123", "atmos")

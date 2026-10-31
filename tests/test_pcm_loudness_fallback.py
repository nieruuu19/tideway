"""Loudness leveling for downloaded (tag-less) PCM files.

A TIDAL stream carries ReplayGain tags; the downloaded copy of the
same track does not, because the downloader muxes audio only. Before
the measured fallback, a downloaded master played at its own level —
"As It Was" (~-6 LUFS) sat ~10 dB louder in Tideway than the same song
in Apple Music, which levels every track server-side.

These pin the fallback: cached measurement applied at load, a
background scan for the first play, tags still winning when present,
and no measurement for streams (their tags are the source of truth).
"""
from __future__ import annotations

import time

import pytest

from app.audio import player as player_mod
from app.audio.player import PCMPlayer, StreamInfo


def _player() -> PCMPlayer:
    return PCMPlayer(lambda: None)


def _local(p: PCMPlayer, path: str, **tags) -> None:
    p._source_path = path
    p._current_stream_info = StreamInfo(source="local", codec="flac", **tags)


def test_cached_measurement_levels_untagged_local_file(monkeypatch):
    p = _player()
    monkeypatch.setattr(
        player_mod.loudness, "cached", lambda path: {"lufs": -6.0, "peak": 1.0}
    )
    _local(p, "/music/As It Was.flac")

    p.set_replaygain("track", 0.0, True)

    state = p.replaygain_state()
    assert state["source"] == "measured"
    assert state["applied_db"] == pytest.approx(-10.0)
    assert state["loudness_lufs"] == pytest.approx(-6.0)


def test_first_play_scans_and_applies_when_measured(monkeypatch):
    p = _player()
    monkeypatch.setattr(player_mod.loudness, "cached", lambda path: None)
    monkeypatch.setattr(
        player_mod.loudness, "analyze", lambda path: {"lufs": -20.0, "peak": 0.5}
    )
    _local(p, "/music/quiet.flac")

    p.set_replaygain("track", 0.0, True)

    # Unmeasured at first: flat, then the scan lands.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if p.replaygain_state()["source"] == "measured":
            break
        time.sleep(0.02)
    state = p.replaygain_state()
    assert state["source"] == "measured"
    # -16 target against a -20 LUFS measure, and a 0.5 peak leaves room.
    assert state["applied_db"] == pytest.approx(4.0)


def test_tags_win_over_measurement(monkeypatch):
    p = _player()

    def _boom(path):
        raise AssertionError("must not measure a tagged file")

    monkeypatch.setattr(player_mod.loudness, "cached", _boom)
    monkeypatch.setattr(player_mod.loudness, "analyze", _boom)
    _local(p, "/music/tagged.flac", track_replay_gain_db=-7.5)

    p.set_replaygain("track", 0.0, True)

    state = p.replaygain_state()
    assert state["source"] == "tags"
    assert state["applied_db"] == pytest.approx(-7.5)


def test_stream_without_tags_is_not_measured(monkeypatch):
    p = _player()

    def _boom(path):
        raise AssertionError("streams have no local file to measure")

    monkeypatch.setattr(player_mod.loudness, "cached", _boom)
    p._source_path = None
    p._current_stream_info = StreamInfo(source="stream", codec="flac")

    p.set_replaygain("track", 0.0, True)

    state = p.replaygain_state()
    assert state["source"] == "none"
    assert state["applied_db"] == 0.0


def test_mode_off_never_measures(monkeypatch):
    p = _player()

    def _boom(path):
        raise AssertionError("mode off must stay bit-perfect")

    monkeypatch.setattr(player_mod.loudness, "cached", _boom)
    _local(p, "/music/x.flac")

    p.set_replaygain("off", 0.0, True)

    assert p.replaygain_state()["source"] == "none"


def test_late_measurement_does_not_touch_a_different_track(monkeypatch):
    p = _player()
    monkeypatch.setattr(
        player_mod.loudness, "cached", lambda path: {"lufs": -12.0, "peak": 0.5}
    )
    _local(p, "/music/playing.flac")
    p.set_replaygain("track", 0.0, True)
    before = p.replaygain_state()
    assert before["source"] == "measured"

    p._apply_measured_pcm_gain("/music/other.flac", {"lufs": -6.0, "peak": 1.0})

    after = p.replaygain_state()
    assert after["source"] == "measured"
    assert after["applied_db"] == before["applied_db"]

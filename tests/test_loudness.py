"""Loudness meter + measured-gain resolution.

The meter has one job that must be exactly right: a known signal has
to read its known loudness. These tests pin the BS.1770 calibration
(a 997 Hz sine at -23 dBFS reads -23 LUFS), the LFE exclusion, the
gating rules, and the cache — the native Atmos path applies whatever
this returns straight to the output level.
"""
from __future__ import annotations

import math
import wave

import numpy as np
import pytest

from app.audio import loudness
from app.audio.replaygain import compute_measured_gain_db

RATE = 48000


def _write_wav(path, data: np.ndarray) -> str:
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(pcm.shape[1])
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes(pcm.tobytes())
    return str(path)


def _sine(rms_db: float, seconds: float, peak_amp: float | None = None) -> np.ndarray:
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    amp = math.sqrt(2.0) * 10 ** (rms_db / 20.0)
    if peak_amp is not None:
        amp = peak_amp
    return amp * np.sin(2 * np.pi * 997 * t)


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loudness, "user_data_dir", lambda: tmp_path)
    return tmp_path


def test_mono_sine_reads_its_loudness(cache_dir, tmp_path):
    path = _write_wav(tmp_path / "mono.wav", _sine(-23.0, 3.0)[:, None])

    result = loudness.measure_file(path)

    assert result["lufs"] == pytest.approx(-23.0, abs=0.15)


def test_second_channel_adds_three_db(cache_dir, tmp_path):
    mono = _sine(-23.0, 3.0)[:, None]
    path = _write_wav(tmp_path / "stereo.wav", np.repeat(mono, 2, axis=1))

    result = loudness.measure_file(path)

    assert result["lufs"] == pytest.approx(-20.0, abs=0.15)


def test_lfe_is_excluded_from_surround_layouts():
    # 5.1(side) / 7.1 put the LFE fourth; its energy must not count.
    assert loudness._channel_weights("5.1(side)", 6)[3] == 0.0
    assert loudness._channel_weights("5.1", 6)[3] == 0.0
    assert loudness._channel_weights("7.1", 8)[3] == 0.0
    assert loudness._channel_weights("stereo", 2).tolist() == [1.0, 1.0]
    # An unnamed layout can't be mapped to a speaker order, so nothing
    # is zeroed rather than guessing the wrong channel.
    assert loudness._channel_weights("6 channels", 6).tolist() == [1.0] * 6


def test_quiet_passage_below_relative_gate_is_ignored(cache_dir, tmp_path):
    loud = _sine(-23.0, 3.0)
    quiet = _sine(-43.0, 3.0)
    path = _write_wav(tmp_path / "gated.wav", np.concatenate([loud, quiet])[:, None])

    result = loudness.measure_file(path)

    # The -43 dBFS half sits 20 LU below the -23 LUFS blocks, so the
    # relative gate drops it and the integrated value stays at the
    # loud passage instead of dragging down by 3 dB.
    assert result["lufs"] == pytest.approx(-23.0, abs=0.3)


def test_peak_is_reported_for_clipping_prevention(cache_dir, tmp_path):
    mono = _sine(-20.0, 2.0, peak_amp=0.5)[:, None]
    path = _write_wav(tmp_path / "peaks.wav", mono)

    result = loudness.measure_file(path)

    assert result["peak"] == pytest.approx(0.5, abs=0.01)


def test_measured_value_is_cached_by_file_version(cache_dir, tmp_path):
    path = _write_wav(tmp_path / "cached.wav", _sine(-18.0, 2.0)[:, None])
    first = loudness.measure_file(path)

    hit = loudness.cached(path)

    assert hit is not None
    assert hit["lufs"] == pytest.approx(first["lufs"])

    # A replaced file must not inherit the old number.
    _write_wav(tmp_path / "cached.wav", _sine(-30.0, 2.0)[:, None])
    assert loudness.cached(path) is None


def test_analyze_reuses_cache(cache_dir, tmp_path, monkeypatch):
    path = _write_wav(tmp_path / "reuse.wav", _sine(-18.0, 2.0)[:, None])
    calls = {"n": 0}
    real = loudness._analyze_streams

    def counted(p):
        calls["n"] += 1
        return real(p)

    monkeypatch.setattr(loudness, "_analyze_streams", counted)

    loudness.analyze(path)
    loudness.analyze(path)

    assert calls["n"] == 1


def test_gain_to_target():
    assert loudness.gain_to_target(-12.0) == pytest.approx(-4.0)
    assert loudness.gain_to_target(-20.0) == pytest.approx(4.0)


def test_measured_gain_without_data_is_flat():
    assert compute_measured_gain_db(None, None, -16.0, 0.0, True) == 0.0


def test_measured_gain_targets_lufs_and_adds_preamp():
    gain = compute_measured_gain_db(-12.0, None, -16.0, 1.5, True)
    assert gain == pytest.approx(-2.5)


def test_measured_gain_respects_peak_ceiling():
    # A track that needs +4 dB but already peaks at full scale can't
    # take any positive gain.
    gain = compute_measured_gain_db(-20.0, 1.0, -16.0, 0.0, True)
    assert gain == pytest.approx(0.0)
    # With clipping prevention off the user gets the boost they asked for.
    assert compute_measured_gain_db(-20.0, 1.0, -16.0, 0.0, False) == pytest.approx(4.0)

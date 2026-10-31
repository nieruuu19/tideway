"""Atmos test panel: capabilities gate and native-engine lifecycle.

The panel's audio engines are faked here — the point is that a
missing helper fails loudly, an unknown engine is rejected, and a
running test reports the decoder modules the helper loaded.
"""
from __future__ import annotations

import pytest

from app.audio import atmos_test as atmos_test_mod
from app.audio.atmos_test import AtmosTester


def _probe_payload() -> dict:
    return {
        "path": r"C:\music\track.m4a",
        "exists": True,
        "size_bytes": 1024,
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration_s": 10.0,
        "audio": {
            "codec": "eac3",
            "profile": "Dolby Digital Plus + Dolby Atmos",
            "channels": 6,
            "layout": "5.1(side)",
            "sample_rate_hz": 48000,
            "bit_rate": 768000,
            "stream_index": 0,
            "audio_streams": 1,
            "duration_s": 10.0,
            "is_atmos": True,
            "atmos_confidence": "container",
        },
        "notes": [],
    }


class _FakeNative:
    def __init__(self) -> None:
        self.closed = False
        self.volume: float | None = None
        self.running = True

    def load(
        self, path: str, autoplay: bool = True, adaptive: bool = False
    ) -> dict:
        return {}

    def set_volume(self, percent: float) -> None:
        self.volume = percent

    def state(self) -> dict:
        return {
            "ok": True,
            "pos_ms": 1234,
            "dur_ms": 10000,
            "state": "Playing",
            "decoders": ["DolbyDDPDecMft.dll"],
            "device": "Speakers (Realtek(R) Audio)",
        }

    def close(self) -> None:
        self.closed = True


def test_capabilities_reports_missing_helper(monkeypatch):
    monkeypatch.setattr(atmos_test_mod, "supported", lambda: False)
    caps = AtmosTester.capabilities()
    windows = next(e for e in caps["engines"] if e["id"] == "windows")
    assert windows["available"] is False
    assert windows["reason"]


def test_play_rejects_unknown_engine():
    with pytest.raises(ValueError):
        AtmosTester().play("x.m4a", "wasapi")


def test_play_windows_requires_helper(monkeypatch):
    monkeypatch.setattr(atmos_test_mod, "probe_media_file", lambda p: _probe_payload())
    monkeypatch.setattr(atmos_test_mod, "supported", lambda: False)
    with pytest.raises(atmos_test_mod.NativeAtmosError):
        AtmosTester().play("x.m4a", "windows")


def test_windows_play_reports_decoder_modules(monkeypatch):
    fake = _FakeNative()
    monkeypatch.setattr(atmos_test_mod, "probe_media_file", lambda p: _probe_payload())
    monkeypatch.setattr(atmos_test_mod, "supported", lambda: True)
    monkeypatch.setattr(
        atmos_test_mod, "NativeAtmosPlayer", lambda: fake
    )

    tester = AtmosTester(volume_getter=lambda: 42)
    status = tester.play(r"C:\music\track.m4a", "windows")

    assert status["engine"] == "windows"
    assert status["state"] == "playing"
    assert status["detail"]["decoders"] == ["DolbyDDPDecMft.dll"]
    assert status["detail"]["device"].startswith("Speakers")
    assert fake.volume == 42

    stopped = tester.stop()
    assert stopped["engine"] is None
    assert fake.closed is True


def test_status_is_idle_before_any_play():
    status = AtmosTester().status()
    assert status["engine"] is None
    assert status["state"] == "idle"

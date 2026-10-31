"""A/B playback tester for Dolby Atmos files.

The point of this panel is to answer "did it actually decode as Atmos,
or only as surround?" with evidence rather than a guess. Two engines
play the same file:

  * `windows` — Windows' Media Foundation pipeline through
    `NativeAtmosPlayer`. Dolby's DD+ decoder MFT renders the JOC
    objects and the endpoint's spatial renderer virtualises them.
  * `tideway` — the normal `PCMPlayer` (PyAV + sounddevice), which
    decodes the core bed and skips the object layer entirely.

Both are isolated from the main player: the tester owns its own
`PCMPlayer` instance and its own `NativeAtmosPlayer`, so a test never
disturbs the queue, the now-playing snapshot, or a live stream. The
caller is expected to stop the main player before starting a test —
two engines writing to the same endpoint at once is exactly the
confusion this panel exists to remove.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from app.audio.atmos_probe import probe_media_file
from app.audio.native_atmos import NativeAtmosPlayer, NativeAtmosError, supported
from app.audio.player import PCMPlayer

_TEST_TRACK_ID = "atmos-test"
_ENGINES = ("windows", "tideway")


class AtmosTester:
    def __init__(self, volume_getter: Optional[Callable[[], int]] = None) -> None:
        self._volume_getter = volume_getter or (lambda: 100)
        self._lock = threading.Lock()
        self._engine: Optional[str] = None
        self._path: Optional[str] = None
        # Probe duration, kept as a fallback: a DASH-fragmented MP4
        # has no top-level duration for mutagen to read, so the PCM
        # snapshot reports 0 even while playback advances.
        self._duration_s: Optional[float] = None
        self._native: Optional[NativeAtmosPlayer] = None
        self._pcm: Optional[PCMPlayer] = None

    # --- capabilities -------------------------------------------------

    @staticmethod
    def capabilities() -> dict:
        native_ok = supported()
        if native_ok:
            native_reason = None
        elif not native_ok:
            native_reason = (
                "Native playback needs Windows, PowerShell and the "
                "bundled winrt_player.ps1 helper."
            )
        return {
            "engines": [
                {
                    "id": "windows",
                    "label": "Windows (Media Foundation)",
                    "available": native_ok,
                    "reason": native_reason,
                    "hint": (
                        "MediaPlayer + Dolby's decoder MFT + the spatial "
                        "renderer. This is the path Windows Media Player "
                        "uses, and the only one that can render height "
                        "objects."
                    ),
                },
                {
                    "id": "tideway",
                    "label": "Tideway (PyAV)",
                    "available": True,
                    "reason": None,
                    "hint": (
                        "The normal PCM engine: E-AC-3 core bed through "
                        "PortAudio. Good for hearing what Tideway does "
                        "without the OS pipeline."
                    ),
                },
            ]
        }

    # --- lifecycle ----------------------------------------------------

    def play(self, path: str, engine: str) -> dict:
        if engine not in _ENGINES:
            raise ValueError(f"unknown engine: {engine}")
        # Probe before touching audio so a bad path returns a clear
        # error and never leaves the previous test half-torn-down.
        probe = probe_media_file(path)
        if engine == "windows" and not supported():
            raise NativeAtmosError(
                "Native playback needs Windows, PowerShell and the "
                "bundled media helper."
            )

        self.stop()
        volume = max(0, min(100, int(self._volume_getter())))
        if engine == "windows":
            native = NativeAtmosPlayer()
            try:
                native.load(probe["path"], autoplay=True)
                native.set_volume(volume)
            except Exception:
                native.close()
                raise
            with self._lock:
                self._native = native
                self._engine = engine
                self._path = probe["path"]
                self._duration_s = probe["duration_s"]
            return self.status()

        pcm = PCMPlayer(
            session_getter=lambda: None,  # local file only
            local_lookup=lambda tid: probe["path"] if tid == _TEST_TRACK_ID else None,
            native_atmos_enabled=lambda: False,
        )
        try:
            pcm.load(_TEST_TRACK_ID)
            pcm.set_volume(volume)
            pcm.play()
        except Exception:
            try:
                pcm.stop()
            except Exception:
                pass
            raise
        with self._lock:
            self._pcm = pcm
            self._engine = engine
            self._path = probe["path"]
            self._duration_s = probe["duration_s"]
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            native, pcm = self._native, self._pcm
            self._native = None
            self._pcm = None
            self._engine = None
            self._path = None
            self._duration_s = None
        if native is not None:
            try:
                native.close()
            except Exception:
                pass
        if pcm is not None:
            try:
                pcm.stop()
            except Exception:
                pass
        return self.status()

    # --- status -------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            engine = self._engine
            native = self._native
            pcm = self._pcm
            path = self._path
            duration_s = self._duration_s
            error = None
        base: dict = {
            "engine": engine,
            "path": path,
            "error": error,
            "state": "idle",
            "position_ms": 0,
            "duration_ms": 0,
            "detail": {},
        }
        if engine == "windows" and native is not None:
            try:
                reply = native.state()
            except Exception as exc:
                base["error"] = str(exc)
                base["state"] = "error"
                return base
            base.update(
                {
                    "state": str(reply.get("state") or "").lower() or "idle",
                    "position_ms": int(reply.get("pos_ms") or 0),
                    "duration_ms": int(reply.get("dur_ms") or 0),
                    "detail": {
                        "decoders": list(reply.get("decoders") or []),
                        "device": reply.get("device") or "",
                        "helper_running": native.running,
                    },
                }
            )
        elif engine == "tideway" and pcm is not None:
            snap = pcm.snapshot()
            info = snap.stream_info
            base.update(
                {
                    "state": snap.state,
                    "position_ms": snap.position_ms,
                    "duration_ms": snap.duration_ms
                    or int((duration_s or 0) * 1000),
                    "error": snap.error,
                    "detail": {
                        "output": pcm.last_open_diagnostics(),
                        "stream": {
                            "codec": getattr(info, "codec", None),
                            "audio_mode": getattr(info, "audio_mode", None),
                            "sample_rate_hz": getattr(
                                info, "sample_rate_hz", None
                            ),
                        },
                    },
                }
            )
        return base

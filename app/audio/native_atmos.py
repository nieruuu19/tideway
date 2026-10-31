"""Native Dolby Atmos playback through Windows' own media pipeline.

Tideway's PCM engine decodes with PyAV, which renders the E-AC-3 core
bed but drops the JOC object metadata that makes Atmos "Atmos". Windows
Media Player sounds different because it doesn't use PyAV: Media
Foundation loads Dolby's decoder MFT, and the decoded object audio goes
to the endpoint's spatial renderer ("Dolby Atmos for Headphones"), which
virtualises height and overhead positions.

This module hosts that same pipeline — `Windows.Media.Playback.
MediaPlayer`, the engine the Media Player app itself uses — inside a
PowerShell helper (`winrt_player.ps1`) and exposes it as a small,
thread-safe player with a line protocol. It is Windows-only. A
downloaded track is given to it as a file; a streamed one is handed
over as a DASH manifest over loopback (`atmos_dash.py`) so playback
starts on the first segment instead of after the whole download.

Everything here is deliberately isolated from `PCMPlayer`. The native
path has no EQ, crossfeed, ReplayGain or gapless handling — those are
the PCM engine's, and an Atmos mix is the one case where the OS
renderer's object stage matters more than Tideway's DSP.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).with_name("winrt_player.ps1")

_REPLY_PREFIX = "@@"
_REQUEST_TIMEOUT_S = 15.0
_START_TIMEOUT_S = 5.0


class NativeAtmosError(RuntimeError):
    """The Windows media pipeline is unavailable or rejected a request."""


def supported() -> bool:
    """Whether this machine can host the native pipeline at all."""
    if not sys.platform.startswith("win"):
        return False
    if not _SCRIPT.exists():
        return False
    return shutil.which("powershell.exe") is not None or shutil.which("powershell") is not None


class NativeAtmosPlayer:
    """One PowerShell-hosted MediaPlayer, driven over stdin/stdout.

    The helper is spawned lazily on the first command and kept alive so
    consecutive tracks reuse one pipeline (MediaPlayer reopens its
    source, which avoids the process-start cost on every track). All
    public methods are thread-safe; a dead helper is respawned on the
    next command rather than surfacing a permanent failure.
    """

    def __init__(self, script: Optional[Path] = None) -> None:
        self._script = script or _SCRIPT
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._replies: "queue.Queue[dict]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._closing = False

    # --- lifecycle ---------------------------------------------------

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def _spawn(self) -> None:
        if not supported():
            raise NativeAtmosError(
                "Native Atmos playback needs Windows and PowerShell."
            )
        exe = shutil.which("powershell.exe") or shutil.which("powershell")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                [
                    exe,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(self._script),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise NativeAtmosError(f"could not start the media helper: {exc}") from exc
        self._proc = proc
        self._replies = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(proc, self._replies),
            name="native-atmos-reader",
            daemon=True,
        )
        self._reader.start()

    @staticmethod
    def _read_loop(proc: subprocess.Popen, replies: "queue.Queue[dict]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if not line.startswith(_REPLY_PREFIX):
                # The Dolby decoder MFT and Media Foundation chatter on
                # the console. Useful when diagnosing, never protocol.
                log.debug("native atmos helper: %s", line)
                continue
            try:
                replies.put(json.loads(line[len(_REPLY_PREFIX):]))
            except ValueError:
                log.debug("native atmos helper: unparseable reply %r", line)
        replies.put({"ok": False, "error": "media helper exited"})

    def _ensure_running(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()

    def close(self) -> None:
        """Stop playback and shut the helper down (best effort)."""
        with self._lock:
            self._closing = True
            proc = self._proc
            self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            self._send(proc, {"cmd": "quit"})
        except Exception:
            pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # --- protocol ----------------------------------------------------

    @staticmethod
    def _send(proc: subprocess.Popen, payload: dict) -> None:
        if proc.stdin is None:
            raise NativeAtmosError("media helper stdin is closed")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def request(self, payload: dict, timeout: float = _REQUEST_TIMEOUT_S) -> dict:
        with self._lock:
            self._closing = False
            self._ensure_running()
            proc = self._proc
            assert proc is not None
            try:
                self._send(proc, payload)
            except (OSError, ValueError) as exc:
                raise NativeAtmosError(
                    f"media helper is not accepting commands: {exc}"
                ) from exc
            try:
                reply = self._replies.get(timeout=timeout)
            except queue.Empty:
                raise NativeAtmosError(
                    f"media helper did not answer {payload.get('cmd')!r} "
                    f"within {timeout:.0f}s"
                ) from None
            if not reply.get("ok", False):
                raise NativeAtmosError(str(reply.get("error") or "media helper error"))
            return reply

    # --- playback ----------------------------------------------------

    def load(self, path: str, autoplay: bool = True, adaptive: bool = False) -> dict:
        # `adaptive` names a DASH manifest URL: the helper builds an
        # AdaptiveMediaSource, the only source kind MediaPlayer will
        # start before the whole track has arrived. A URL carries no
        # file to check, so only a filesystem path is validated here.
        if not path.startswith(("http://", "https://")):
            path = os.path.abspath(path)
            if not os.path.isfile(path):
                raise NativeAtmosError(f"file not found: {path}")
        start = _REQUEST_TIMEOUT_S + _START_TIMEOUT_S
        return self.request(
            {
                "cmd": "load",
                "path": path,
                "autoplay": bool(autoplay),
                "adaptive": bool(adaptive),
            },
            timeout=start,
        )

    def play(self) -> dict:
        return self.request({"cmd": "play"})

    def pause(self) -> dict:
        return self.request({"cmd": "pause"})

    def seek_ms(self, ms: int) -> dict:
        return self.request({"cmd": "seek", "ms": max(0, int(ms))})

    def set_volume(self, percent: float) -> dict:
        return self.request({"cmd": "volume", "value": float(percent)})

    def set_gain_db(self, gain_db: float) -> dict:
        """Loudness-normalization offset, in dB, applied on top of the
        user's volume. The helper composes the two into MediaPlayer's
        single 0..1 volume stage, so the reply's `gain_applied_db`
        is the offset that was actually reachable."""
        return self.request({"cmd": "gain", "value": float(gain_db)})

    def set_muted(self, muted: bool) -> dict:
        return self.request({"cmd": "mute", "value": bool(muted)})

    def state(self) -> dict:
        return self.request({"cmd": "state"}, timeout=5.0)

    def stop(self) -> dict:
        return self.request({"cmd": "stop"})

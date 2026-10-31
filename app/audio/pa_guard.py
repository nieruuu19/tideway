"""Serialization for PortAudio's process-global state.

sounddevice stream handles are raw CFFI pointers and
`OutputStream.close()` is not atomic: it calls `Pa_CloseStream(_ptr)`
and only afterwards clears `_ptr`. Two threads letting go of the same
stream therefore both pass the `closed` check and the loser closes
freed memory — an access violation inside libportaudio64bit. That is
what took the backend down repeatedly right after a Windows Spatial
Sound change, when the finished callback and the recovery thread each
release the same stream (crash dumps: `pcm-device-recovery`,
libportaudio64bit+0xce38, three times).

The same hazard applies to `Pa_Terminate` / `Pa_Initialize`, which
frees and reallocates every session-wide object: running that while a
stream exists leaves that stream's pointer dangling, and running it
concurrently with an open/close races the same structures.

So both operations go through one lock here, and every stream opened
through `open_stream` is tracked so `reinit_portaudio` can refuse to
tear the library down while a stream is still alive.
"""
from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Optional

import sounddevice as sd  # type: ignore

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_STREAMS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def open_stream(**kwargs) -> Any:
    """`sd.OutputStream(**kwargs)`, serialized and tracked."""
    with _LOCK:
        stream = sd.OutputStream(**kwargs)
        _STREAMS.add(stream)
        return stream


def close_stream(stream: Optional[Any]) -> None:
    """Abort + close `stream` at most once, whoever calls it.

    Idempotent and safe to race with another closer: the `closed`
    check and the close itself happen under the same lock.
    """
    if stream is None:
        return
    with _LOCK:
        if not getattr(stream, "closed", True):
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        _STREAMS.discard(stream)


def live_stream_count() -> int:
    with _LOCK:
        return sum(
            1 for stream in _STREAMS if not getattr(stream, "closed", True)
        )


def reinit_portaudio() -> bool:
    """Terminate + initialize PortAudio so endpoints re-enumerate.

    Returns True when the reinit ran, False when it was skipped
    because a stream is still open somewhere in the process (the
    Atmos tester runs its own player, so "no local stream" is not
    enough to know). Callers that need a fresh enumeration should fall
    back to what they do without one — reopening a different host-API
    copy of the same endpoint doesn't need it.
    """
    with _LOCK:
        live = sum(
            1 for stream in _STREAMS if not getattr(stream, "closed", True)
        )
        if live:
            log.warning(
                "PortAudio reinit skipped: %d stream(s) still open", live
            )
            return False
        try:
            sd._terminate()
        except Exception:
            pass
        try:
            sd._initialize()
        except Exception:
            log.exception("PortAudio reinit failed")
        return True

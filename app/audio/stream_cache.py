"""Persistent disk cache for streamed, non-Atmos tracks.

Streamed PCM playback used to re-download a track on every play: libav
reads the DASH segments through `SegmentReader`, and the bytes went
away with the decoder. This cache is a tee on that reader — segments
are appended to a `.part` file in arrival order, renamed to the final
file once the last one lands, and described by a JSON sidecar (size,
duration, StreamInfo, TIDAL metadata). A later resolve opens the file
directly: no TIDAL round-trip, no download, and precise in-track
seeking (a real file, so libav can seek it).

Atmos tracks are not stored here. Their segments are already joined
into `atmos_cache/` by the native pipeline, and that pipeline needs
the object layer this one does not touch.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

from app.paths import user_data_dir

log = logging.getLogger(__name__)

_CACHE_DIRNAME = "stream_cache"
_VERSION = 1

# Budgets. A lossless track is ~30-60 MB, so 2 GB holds ~40 of them;
# lossy tiers fit hundreds. Age is the other axis: a cached track no
# one played for two months is dead weight either way.
_MAX_BYTES = 2 * 1024 * 1024 * 1024
_MAX_AGE_S = 60.0 * 24 * 3600
# A `.part` older than this was left by a process that died: live
# writers keep rewriting theirs while the track downloads.
_PART_MAX_AGE_S = 3600.0


def _safe_name(value: str, limit: int = 64) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value)
    )[:limit]
    return safe or "x"


def _sidecar_path(path: str) -> str:
    return path + ".json"


def _unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _read_sidecar(path: str) -> Optional[dict]:
    """The sidecar's payload, or None when it is missing/unreadable or
    describes a file that changed size (an interrupted rewrite)."""
    try:
        with open(_sidecar_path(path), encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict) or payload.get("version") != _VERSION:
            return None
        if payload.get("size") != os.path.getsize(path):
            return None
        if not isinstance(payload.get("info"), dict):
            return None
    except (OSError, ValueError):
        return None
    return payload


class StreamSink:
    """Append-only writer for one track's segment bytes.

    Segments must arrive in index order (0 = init, 1..N = media). Any
    gap aborts the write — the joined file would not be playable. The
    write is published atomically: `.part` is renamed only after the
    last segment lands and its sidecar written, so a crash or a skip
    never leaves a truncated file that the next play would trust.
    """

    def __init__(
        self,
        final_path: str,
        total: int,
        duration_s: Optional[float],
        info: dict,
        meta: Optional[dict],
    ) -> None:
        self._final = final_path
        self._part = (
            f"{final_path}.{os.getpid()}.{threading.get_ident()}.part"
        )
        self._total = total
        self._duration_s = duration_s
        self._info = info
        self._meta = meta
        self._next = 0
        self._written = 0
        self._done = False
        self._lock = threading.Lock()
        self._fh: Optional[Any] = None
        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            self._fh = open(self._part, "wb")
        except OSError as exc:
            log.debug("stream cache: cannot write %s: %s", self._part, exc)
            self._done = True

    def write(self, index: int, data: bytes) -> None:
        with self._lock:
            if self._done or self._fh is None:
                return
            if index != self._next or not data:
                self._abort_locked()
                return
            try:
                self._fh.write(data)
                self._fh.flush()
            except OSError as exc:
                log.debug("stream cache: write failed for %s: %s", self._part, exc)
                self._abort_locked()
                return
            self._next += 1
            self._written += len(data)
            if self._next >= self._total:
                self._finish_locked()

    def finish(self) -> None:
        """Publish the file if every segment arrived; no-op otherwise."""
        with self._lock:
            if not self._done and self._next >= self._total:
                self._finish_locked()

    def abort(self) -> None:
        """Drop the partial file. Idempotent; safe after `finish`."""
        with self._lock:
            if not self._done:
                self._abort_locked()

    # --- internals -------------------------------------------------

    def _finish_locked(self) -> None:
        self._done = True
        fh, self._fh = self._fh, None
        try:
            if fh is not None:
                fh.close()
        except OSError:
            pass
        try:
            os.replace(self._part, self._final)
            payload = {
                "version": _VERSION,
                "size": self._written,
                "duration_s": self._duration_s,
                "info": self._info,
                "meta": self._meta,
            }
            partial = _sidecar_path(self._final) + ".part"
            with open(partial, "w", encoding="utf-8") as sh:
                json.dump(payload, sh)
            os.replace(partial, _sidecar_path(self._final))
        except (OSError, TypeError, ValueError) as exc:
            # A file without a sidecar is untrustworthy; drop both so
            # the next play downloads instead of opening a bad file.
            log.debug("stream cache: finish failed for %s: %s", self._final, exc)
            _unlink(_sidecar_path(self._final) + ".part")
            _unlink(self._final)
            return
        log.info(
            "stream cache: stored %s (%d bytes, %d segments)",
            self._final,
            self._written,
            self._next,
        )

    def _abort_locked(self) -> None:
        self._done = True
        fh, self._fh = self._fh, None
        try:
            if fh is not None:
                fh.close()
        except OSError:
            pass
        _unlink(self._part)


class StreamCache:
    """Thread-safe (track_id, quality) → finished file cache.

    `StreamInfo` and the TIDAL metadata dict are held opaquely (Any /
    dict) and handed back verbatim; the player is the only reader.
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        *,
        max_bytes: int = _MAX_BYTES,
        max_age_s: float = _MAX_AGE_S,
    ) -> None:
        self._directory_override = directory
        self._dir: Optional[str] = None
        self._max_bytes = max_bytes
        self._max_age_s = max_age_s
        self._swept = False
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def directory(self) -> str:
        """The cache dir, created on demand.

        Resolved once but re-created on every use: the directory can
        be deleted while the app runs, and that must cost a
        re-download, never a failed play.
        """
        pending_sweep = False
        with self._lock:
            if self._dir is None:
                self._dir = self._directory_override or str(
                    self._default_directory()
                )
                pending_sweep = not self._swept
                self._swept = True
            resolved = self._dir
        try:
            os.makedirs(resolved, exist_ok=True)
        except OSError as exc:
            log.debug("stream cache dir unavailable: %s", exc)
        if pending_sweep:
            try:
                self.sweep()
            except Exception:
                log.debug("stream cache sweep failed", exc_info=True)
        return resolved

    @staticmethod
    def _default_directory() -> str:
        override = os.environ.get("TIDEWAY_STREAM_CACHE_DIR")
        if override:
            return override
        return str(user_data_dir() / _CACHE_DIRNAME)

    def path_for(self, track_id: str, quality: Optional[str]) -> str:
        q = _safe_name(quality or "default", limit=24)
        return os.path.join(
            self.directory(), f"{_safe_name(track_id)}-{q}.m4a"
        )

    def lookup(
        self, track_id: str, quality: Optional[str]
    ) -> Optional[tuple[str, Optional[float], dict, Optional[dict]]]:
        """(path, duration_s, info_dict, meta) for a finished file."""
        path = self.path_for(track_id, quality)
        payload = _read_sidecar(path)
        if payload is None:
            with self._lock:
                self._misses += 1
            return None
        duration = payload.get("duration_s")
        meta = payload.get("meta")
        with self._lock:
            self._hits += 1
        return (
            path,
            float(duration) if isinstance(duration, (int, float)) else None,
            payload["info"],
            meta if isinstance(meta, dict) else None,
        )

    def sink(
        self,
        track_id: str,
        quality: Optional[str],
        total_segments: int,
        duration_s: Optional[float],
        info: dict,
        meta: Optional[dict],
    ) -> StreamSink:
        return StreamSink(
            self.path_for(track_id, quality),
            total_segments,
            duration_s,
            info,
            meta,
        )

    def sweep(self) -> None:
        """Remove debris and hold the age/size budget, oldest first."""
        with self._lock:
            resolved = self._dir
        if not resolved:
            return
        try:
            names = os.listdir(resolved)
        except OSError:
            return
        now = time.time()
        files: list[tuple[float, str, int]] = []
        for name in names:
            path = os.path.join(resolved, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if name.endswith(".part"):
                if now - st.st_mtime > _PART_MAX_AGE_S:
                    _unlink(path)
                continue
            if name.endswith(".json"):
                if not os.path.exists(path[:-5]):
                    _unlink(path)
                continue
            if not name.endswith(".m4a"):
                continue
            if not os.path.exists(_sidecar_path(path)):
                # A finished file always has a sidecar; one this old
                # without is a write that died before finishing.
                if now - st.st_mtime > _PART_MAX_AGE_S:
                    _unlink(path)
                continue
            if now - st.st_mtime > self._max_age_s:
                _unlink(path)
                _unlink(_sidecar_path(path))
                continue
            files.append((st.st_mtime, path, st.st_size))
        total = sum(size for _mtime, _path, size in files)
        for _mtime, path, size in sorted(files):
            if total <= self._max_bytes:
                break
            _unlink(path)
            _unlink(_sidecar_path(path))
            total -= size

    def stats(self) -> dict:
        with self._lock:
            resolved = self._dir
            hits = self._hits
            misses = self._misses
        entries = []
        total = 0
        if resolved:
            try:
                names = os.listdir(resolved)
            except OSError:
                names = []
            for name in names:
                if not name.endswith(".m4a"):
                    continue
                path = os.path.join(resolved, name)
                if not os.path.exists(_sidecar_path(path)):
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                total += size
                entries.append({"name": name, "bytes": size})
        return {
            "hits": hits,
            "misses": misses,
            "size": len(entries),
            "bytes_cached": total,
            "bytes_cap": self._max_bytes,
            "entries": entries,
        }

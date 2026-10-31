"""EBU R128 integrated loudness, measured locally.

Sound Check levels every track to one reference so a compressed 2020s
pop master and a quiet 1985 CD rip land at the same perceived volume.
Apple measures that server-side and ships the offset with the track;
TIDAL does the same for stereo through ReplayGain tags, but a
downloaded Atmos file carries no tags at all (the downloader muxes the
manifest's audio and metadata, not TIDAL's normalization facts). So
the native Atmos path has no offset to apply unless we measure the
file ourselves.

This module is that measurement: ITU-R BS.1770-4 / EBU R128 gated
integrated loudness + sample peak, decoded with PyAV. It serves both
tag-less paths: the native Atmos pipeline, and downloaded stereo files
(which also carry no tags, unlike the streamed version of the same
track, so the PCM path measures them too).

The analysis is a full decode, so it lives behind a cache keyed by
path + size + mtime. A second play of the same file is a dict lookup.
Cache writes are atomic (tmp + rename) because the app can be killed
at any moment and a half-written JSON would disable normalization
until the file is deleted by hand.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import av
import numpy as np
from scipy.signal import lfilter

from app.paths import user_data_dir

log = logging.getLogger(__name__)

# Apple's Sound Check levels tracks to roughly -16 LUFS — which is why
# As It Was (a ~-6 LUFS master) measured -10.2 dB in the comparison
# that motivated this module. Matching that target is what makes
# "leveling" mean the same thing here as in the player it's modelled
# on.
TARGET_LUFS = -16.0

# BS.1770 gating thresholds.
_ABS_GATE_LUFS = -70.0
_REL_GATE_LU = -10.0
_SAMPLE_RATE = 48000
_BLOCK_S = 0.4
_HOP_S = 0.1

# K-weighting for 48 kHz, ITU-R BS.1770-4 table 1/2. The shelf stage
# models the head's acoustic transfer, the high-pass removes sub-bass
# energy that doesn't contribute to perceived loudness. Coefficients
# are rate-specific, which is why every stream is resampled first.
_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_HIGHPASS_B = np.array([1.0, -2.0, 1.0])
_HIGHPASS_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

_CACHE_NAME = "loudness_cache.json"
_CACHE_MAX_ENTRIES = 4000

_CACHE_LOCK = threading.Lock()


def _cache_path() -> Path:
    return user_data_dir() / _CACHE_NAME


def _cache_key(path: str) -> Optional[str]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"


def _read_cache() -> dict:
    try:
        with _cache_path().open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(cache: dict) -> None:
    path = _cache_path()
    tmp = path.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        os.replace(tmp, path)
    except OSError as exc:
        # Losing the cache costs one re-measure on the next play; it
        # must never take playback down.
        log.debug("loudness cache write failed: %s", exc)
        try:
            tmp.unlink()
        except OSError:
            pass


def cached(path: str) -> Optional[dict]:
    """Previously measured `{lufs, peak}` for this exact file version."""
    key = _cache_key(path)
    if key is None:
        return None
    with _CACHE_LOCK:
        entry = _read_cache().get(key)
    if not isinstance(entry, dict):
        return None
    lufs = entry.get("lufs")
    if not isinstance(lufs, (int, float)):
        return None
    peak = entry.get("peak")
    return {
        "lufs": float(lufs),
        "peak": float(peak) if isinstance(peak, (int, float)) else None,
    }


def _store(path: str, lufs: float, peak: Optional[float]) -> None:
    key = _cache_key(path)
    if key is None:
        return
    with _CACHE_LOCK:
        cache = _read_cache()
        cache[key] = {
            "lufs": lufs,
            "peak": peak,
            "analyzed_at": int(time.time()),
        }
        if len(cache) > _CACHE_MAX_ENTRIES:
            keep = sorted(
                cache.items(),
                key=lambda kv: kv[1].get("analyzed_at", 0)
                if isinstance(kv[1], dict)
                else 0,
            )[-_CACHE_MAX_ENTRIES:]
            cache = dict(keep)
        _write_cache(cache)


def _channel_weights(layout_name: str, channels: int) -> np.ndarray:
    """BS.1770 channel gains. LFE is excluded: it carries no
    perceived loudness, only headroom cost. Everything else is 1.0
    (BS.1770-4 leaves surround channels unweighted; the 1.41 surround
    gain of the original spec was dropped)."""
    weights = np.ones(channels, dtype=np.float64)
    lfe_index = None
    low = (layout_name or "").lower()
    if low.startswith("5.1"):
        lfe_index = 3
    elif low.startswith("7.1"):
        lfe_index = 3
    elif low.startswith("6.1"):
        lfe_index = 3
    if lfe_index is not None and channels > lfe_index:
        weights[lfe_index] = 0.0
    return weights


def _k_weight(x: np.ndarray) -> np.ndarray:
    y = lfilter(_SHELF_B, _SHELF_A, x)
    return lfilter(_HIGHPASS_B, _HIGHPASS_A, y)


def _block_energies(energy: np.ndarray) -> np.ndarray:
    """400 ms blocks at 100 ms hops, as mean square per block.

    Summing the K-weighted power of every channel first and windowing
    afterwards is equivalent to windowing each channel and summing —
    and keeps one float64 array in flight instead of one per channel.
    """
    hop = int(round(_HOP_S * _SAMPLE_RATE))
    per_block = int(round(_BLOCK_S / _HOP_S))
    hops = len(energy) // hop
    if hops < per_block:
        return np.empty(0, dtype=np.float64)
    segments = energy[: hops * hop].reshape(hops, hop).sum(axis=1) / hop
    windowed = np.convolve(segments, np.ones(per_block), mode="valid")
    return windowed / per_block


def _integrated_lufs(block_energies: np.ndarray) -> Optional[float]:
    if block_energies.size == 0:
        return None
    with np.errstate(divide="ignore"):
        loudness = -0.691 + 10.0 * np.log10(block_energies)
    above_abs = loudness > _ABS_GATE_LUFS
    if not np.any(above_abs):
        return None
    abs_gated = block_energies[above_abs]
    relative = -0.691 + 10.0 * math.log10(float(np.mean(abs_gated)))
    above_rel = above_abs & (loudness > relative + _REL_GATE_LU)
    if not np.any(above_rel):
        return None
    gated = block_energies[above_rel]
    return -0.691 + 10.0 * math.log10(float(np.mean(gated)))


def _analyze_streams(path: str) -> dict:
    """Decode the first audio stream and return loudness + peak."""
    with av.open(path) as container:
        streams = list(container.streams.audio)
        if not streams:
            raise ValueError("file has no audio stream")
        stream = streams[0]
        layout_name = getattr(getattr(stream, "layout", None), "name", "") or ""
        resampler = av.AudioResampler(
            format="fltp",
            layout=stream.layout.name,
            rate=_SAMPLE_RATE,
        )
        weights: Optional[np.ndarray] = None
        parts: list[np.ndarray] = []
        peak = 0.0

        def _consume(frame) -> None:
            nonlocal weights, peak
            arr = frame.to_ndarray()
            # fltp comes back as (channels, samples).
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            x = np.ascontiguousarray(arr, dtype=np.float64)
            if weights is None or weights.shape[0] != x.shape[0]:
                weights = _channel_weights(layout_name, x.shape[0])
            peak = max(peak, float(np.max(np.abs(x))) if x.size else 0.0)
            for ch in range(x.shape[0]):
                x[ch] = _k_weight(x[ch])
            parts.append((x * x).T @ weights)

        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                _consume(out)
        for out in resampler.resample(None):
            _consume(out)

    energy = (
        np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    )
    lufs = _integrated_lufs(_block_energies(energy))
    return {"lufs": lufs, "peak": peak or None}


def measure_file(path: str) -> dict:
    """Measure `path` without consulting the cache. Raises
    FileNotFoundError / ValueError when the file can't be opened or
    has no audio; a track with no measurable loudness (silence) comes
    back with `lufs = None`."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    result = _analyze_streams(path)
    lufs = result.get("lufs")
    if lufs is not None:
        _store(path, float(lufs), result.get("peak"))
    return result


def analyze(path: str) -> Optional[dict]:
    """Cached-or-measured loudness. None when the file can't be read;
    a silent file returns `{"lufs": None, "peak": ...}`."""
    hit = cached(path)
    if hit is not None:
        return hit
    try:
        return measure_file(path)
    except (OSError, ValueError, av.FFmpegError) as exc:
        log.debug("loudness analysis failed for %s: %s", path, exc)
        return None


def gain_to_target(
    lufs: float,
    target_lufs: float = TARGET_LUFS,
) -> float:
    """Offset that brings `lufs` to the target. Negative attenuates
    a loud master, positive lifts a quiet one."""
    return float(target_lufs) - float(lufs)

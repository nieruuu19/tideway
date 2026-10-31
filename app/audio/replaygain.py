"""ReplayGain loudness leveling for the PCM engine.

Cross-album (and cross-track) volume jumps are the audiophile-
community's most-cited reason to reach for the volume knob mid-listen.
Tidal masters come pre-tagged with EBU R128 ReplayGain values + actual
sample peaks; this module applies them at playback time so a quiet
indie record and a brick-walled pop track land at roughly the same
perceived loudness.

Two modes:

* **track** — apply the per-track gain. Best for shuffle / mixed-
  source playback where each track's loudness is independent.
* **album** — apply the album-wide gain (all tracks on an album get
  the same offset). Best for album-as-album listening; preserves the
  artist-intended dynamic range between tracks within the album.

Both are gated by an opt-in user setting. Default is **off** so the
audio path stays bit-perfect for users who haven't asked for leveling.

Clipping prevention: a +6 dB ReplayGain offset on a track that
already peaks at -3 dBFS would clip. When `prevent_clipping` is on
(default), the applied gain is clamped so peak * gain ≤ 1.0. Users
who want loud + risk-of-clip can turn it off.

Implementation: a single linear scalar multiply against the float32
audio buffer in the callback. Cheap enough that we fold it into the
EQ/Crossfeed int↔float round-trip when those are also active. When
RG is the only active stage, we still pay one round-trip to apply
it correctly to int PCM.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Literal, Optional


# Mode strings persisted to settings + read at the API boundary.
ReplayGainMode = Literal["off", "track", "album"]
VALID_MODES: tuple[str, ...] = ("off", "track", "album")

# Reference level TIDAL normalizes its own ReplayGain tags against.
# Measured against a downloaded copy of "As It Was": TIDAL ships
# trackReplayGain = -7.76 dB for a master that our R128 meter puts at
# -5.86 LUFS, i.e. their gain aims for about -14 LUFS, not Apple's
# -16. Everything else Tideway levels (downloaded files, Atmos) targets
# loudness.TARGET_LUFS, so streamed tags need this offset to land on
# the same reference.
TIDAL_TAG_REFERENCE_LUFS = -14.0


@dataclass
class ReplayGainTags:
    """The four numbers ReplayGain needs from a Tidal stream. Held
    as an explicit dataclass instead of a tuple so callers don't have
    to remember which slot is which — the StreamInfo → ReplayGain
    handoff is a hot path and a wrong index would silently apply the
    track gain with the album peak (or vice versa)."""

    track_gain_db: Optional[float] = None
    track_peak: Optional[float] = None
    album_gain_db: Optional[float] = None
    album_peak: Optional[float] = None


def _clamp_to_peak(total: float, peak: Optional[float], prevent_clipping: bool) -> float:
    """Cap `total` so peak * 10**(total/20) ≤ 1.0 when clipping
    prevention is on. For peaks already above unity the cap is
    negative — those masters always attenuate."""
    if prevent_clipping and peak is not None and peak > 0:
        max_gain = -20.0 * math.log10(float(peak))
        if total > max_gain:
            return max_gain
    return total


def compute_gain_db(
    tags: ReplayGainTags,
    mode: ReplayGainMode,
    preamp_db: float,
    prevent_clipping: bool,
    reference_shift_db: float = 0.0,
) -> float:
    """Resolve the dB offset to apply for the active mode.

    Returns 0.0 (no change) when the mode is "off" or when the chosen
    tag is missing on this stream — falling back to flat output is the
    only sane "we don't have data" behaviour. The user's `preamp_db`
    is added on top of the ReplayGain value, plus `reference_shift_db`
    when the tags come from a source with a different reference level
    (TIDAL's own tags aim at -14 LUFS; see TIDAL_TAG_REFERENCE_LUFS).
    Clipping prevention then clamps the total against the chosen peak.
    """
    if mode == "off":
        return 0.0
    if mode == "track":
        rg = tags.track_gain_db
        peak = tags.track_peak
    else:  # "album"
        rg = tags.album_gain_db
        peak = tags.album_peak
    if rg is None:
        # No tag — fall back to flat. Mode itself stays selected so
        # the next track's tag is read; this is a per-stream skip,
        # not a feature toggle.
        return 0.0
    total = float(rg) + float(preamp_db) + float(reference_shift_db)
    return _clamp_to_peak(total, peak, prevent_clipping)


def compute_measured_gain_db(
    loudness_lufs: Optional[float],
    peak: Optional[float],
    target_lufs: float,
    preamp_db: float,
    prevent_clipping: bool,
) -> float:
    """Same contract as `compute_gain_db`, but the loudness comes from
    a local EBU R128 measurement instead of shipped tags. Used on the
    native Atmos path, whose files carry no ReplayGain tags at all.

    Returns 0.0 when there is nothing to measure against (silence, or
    a file the meter couldn't read) so the track plays at its own
    level rather than at a guess.
    """
    if loudness_lufs is None:
        return 0.0
    total = (float(target_lufs) - float(loudness_lufs)) + float(preamp_db)
    return _clamp_to_peak(total, peak, prevent_clipping)


class ReplayGain:
    """Stateful single-stage gain multiplier.

    Thread model: `set_gain_db` / `clear` are called from the HTTP
    handler thread and the player's load path; `apply` runs in the
    audio callback. A short lock guards the linear-gain swap so the
    callback never reads a partially-updated value.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gain_linear: float = 1.0

    def set_gain_db(self, gain_db: float) -> None:
        """Install a new gain. 0 dB resets to unity (effectively
        bypass without taking the lock on every callback)."""
        if gain_db == 0.0:
            self.clear()
            return
        linear = 10.0 ** (float(gain_db) / 20.0)
        with self._lock:
            self._gain_linear = linear

    def clear(self) -> None:
        with self._lock:
            self._gain_linear = 1.0

    def is_active(self) -> bool:
        return self._gain_linear != 1.0

    def gain_linear(self) -> float:
        with self._lock:
            return self._gain_linear

    def apply(self, samples) -> None:
        """In-place scalar multiply. Float32 buffer expected (caller
        round-trips int PCM through float32 the same way EQ does)."""
        if self._gain_linear == 1.0:
            return
        with self._lock:
            samples *= self._gain_linear

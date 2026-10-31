"""Stereo FLAC (lossless / hi-res) stream resolution for TIDAL.

The v1 `playbackinfopostpaywall` endpoint that tidalapi drives
(`Track.get_stream`) is supposed to serve FLAC for the LOSSLESS and
HI_RES_LOSSLESS tiers, but on some OAuth clients it silently answers
AAC 320 (`HIGH`) for every tier — which is how "Lossless"/"Max" ended
up playing the Low badge. The OpenAPI v2 `trackManifests` endpoint
still serves the real FLAC asset for those same sessions, as a base64
DASH manifest with one `flac` representation.

This module fetches that manifest and expands it into the same
`[init, segment, ...]` URL list the Atmos resolver returns, so the PCM
pipeline (`SegmentReader` → PyAV) plays it unchanged. It also reports
which tier the bytes actually are, so the now-playing badge stays
honest when a Max request lands on a CD-rate master.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from app import atmos as atmos_mod

log = logging.getLogger(__name__)

LOSSLESS_FORMAT = "FLAC"
CD_BIT_DEPTH = 16
CD_SAMPLE_RATE = 48_000


class LosslessUnavailable(RuntimeError):
    """TIDAL has no playable FLAC asset for this track / login.

    Raised for "this track only exists as AAC", "this client isn't
    entitled to the FLAC asset", and DRM-protected manifests. Callers
    treat it as a quality fallback signal, never a fatal error.
    """


@dataclass(frozen=True)
class LosslessStream:
    urls: list[str]
    codec: str = "flac"
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    bandwidth: Optional[int] = None
    representation_id: str = ""
    manifest_hash: Optional[str] = None
    track_replay_gain: Optional[float] = None
    track_peak: Optional[float] = None
    album_replay_gain: Optional[float] = None
    album_peak: Optional[float] = None

    @property
    def is_hi_res(self) -> bool:
        return bool(
            (self.bit_depth or 0) > CD_BIT_DEPTH
            or (self.sample_rate or 0) > CD_SAMPLE_RATE
        )

    @property
    def tier(self) -> str:
        return "hi_res_lossless" if self.is_hi_res else "high_lossless"


def _is_flac(representation: dict) -> bool:
    codec = str(representation.get("codec") or "").lower()
    mime = str(representation.get("mimeType") or "").lower()
    return "flac" in codec or "flac" in mime


def _is_hi_res(representation: dict) -> bool:
    return bool(
        (representation.get("bitDepth") or 0) > CD_BIT_DEPTH
        or (representation.get("sampleRate") or 0) > CD_SAMPLE_RATE
    )


def _fidelity_key(representation: dict) -> tuple:
    return (
        int(representation.get("bitDepth") or 0),
        int(representation.get("sampleRate") or 0),
        int(representation.get("bandwidth") or 0),
        int(representation.get("channels") or 0),
    )


def fetch_lossless_stream(
    session: Any, track_id: int, *, hi_res: bool
) -> LosslessStream:
    """Fetch the FLAC asset for one catalog track id.

    `hi_res=True` (Max) picks the best representation TIDAL offers;
    `hi_res=False` (High) prefers the CD-rate one and only falls back
    to a hi-res representation when that's all the track has — still
    far better than the AAC the v1 endpoint hands back.
    """
    attributes: Optional[dict] = None
    last_error: Optional[atmos_mod.AtmosUnavailable] = None
    # PLAYBACK is what a player is entitled to. DOWNLOAD is the
    # fallback for clients/regions where only the download manifest is
    # rendered; the bytes are identical.
    for usage in ("PLAYBACK", "DOWNLOAD"):
        try:
            attributes = atmos_mod.fetch_manifest_attributes(
                session,
                int(track_id),
                usage,
                formats=LOSSLESS_FORMAT,
                label="FLAC",
            )
            break
        except atmos_mod.AtmosUnavailable as exc:
            last_error = exc
            log.debug(
                "FLAC manifest usage=%s unavailable for track %s: %s",
                usage,
                track_id,
                exc,
            )
    if attributes is None:
        raise LosslessUnavailable(
            str(last_error)
            if last_error is not None
            else "TIDAL did not return a FLAC manifest."
        )

    formats = {str(fmt).upper() for fmt in attributes.get("formats") or []}
    if LOSSLESS_FORMAT not in formats:
        raise LosslessUnavailable(
            "TIDAL did not return a FLAC manifest for this track."
        )

    try:
        xml = atmos_mod._decode_manifest_data_uri(attributes.get("uri") or "")
        representations = [
            rep for rep in atmos_mod.parse_dash_audio(xml) if _is_flac(rep)
        ]
    except atmos_mod.AtmosUnavailable as exc:
        raise LosslessUnavailable(str(exc)) from exc
    if not representations:
        raise LosslessUnavailable(
            "TIDAL's FLAC manifest contained no FLAC representation."
        )

    representations.sort(key=_fidelity_key)
    if hi_res:
        representation = representations[-1]
    else:
        cd = [rep for rep in representations if not _is_hi_res(rep)]
        representation = cd[-1] if cd else representations[0]

    track_gain, track_peak = atmos_mod.normalization_facts(attributes, "track")
    album_gain, album_peak = atmos_mod.normalization_facts(attributes, "album")
    return LosslessStream(
        urls=list(representation["urls"]),
        codec=representation.get("codec") or "flac",
        sample_rate=representation.get("sampleRate"),
        channels=representation.get("channels"),
        bit_depth=representation.get("bitDepth"),
        bandwidth=representation.get("bandwidth"),
        representation_id=representation.get("id") or "",
        manifest_hash=attributes.get("hash"),
        track_replay_gain=track_gain,
        track_peak=track_peak,
        album_replay_gain=album_gain,
        album_peak=album_peak,
    )

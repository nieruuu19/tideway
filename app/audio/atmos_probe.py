"""Inspect a local media file for the Dolby Atmos test panel.

The tester answers one question: did this file actually decode as
Atmos, or as plain surround? Two different engines answer it from
different evidence:

  * The Windows engine (`NativeAtmosPlayer`) reports which decoder
    DLLs MediaPlayer loaded — Dolby's DD+ MFT appearing proves the
    OS pipeline decoded it.
  * The Tideway engine is PyAV, which only ever sees the E-AC-3
    core bed. Its evidence is the container's own profile string.

`probe_media_file` is the shared pre-flight: codec, profile,
channel layout and duration, read straight out of the container.
libav reports the JOC object layer as the codec profile
("Dolby Digital Plus + Dolby Atmos"), which is a reliable marker —
more so than scanning the bitstream for a 'joc' magic, which
produced false positives inside media payloads.
"""
from __future__ import annotations

import os
from typing import Any

import av

# Audio codecs whose carriage can include Atmos objects. AC-3 (plain
# Dolby Digital) cannot — only E-AC-3 JOC and AC-4 do.
_ATMOS_CODECS = ("eac3", "ac4")


def _safe_float(value: Any) -> "float | None":
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_media_file(path: str) -> dict:
    """Read container/codec facts for `path`.

    Raises FileNotFoundError / ValueError when the path is missing or
    not something libav can open; the endpoint turns those into 400s.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        size_bytes = os.path.getsize(path)
    except OSError:
        size_bytes = None

    with av.open(path) as container:
        container_name = container.format.name if container.format else None
        duration_s = (
            _safe_float(container.duration) / av.time_base
            if container.duration
            else None
        )
        streams = list(container.streams.audio)
        if not streams:
            raise ValueError("file has no audio stream")
        # The "primary" stream is the widest one — the Atmos bed, when
        # present, is the one with height/back channels.
        primary = max(streams, key=lambda s: s.codec_context.channels or 0)
        ctx = primary.codec_context
        codec = str(ctx.name or "")
        profile = str(ctx.profile or "")
        channels = int(ctx.channels or 0)
        layout = ctx.layout.name if ctx.layout is not None else None
        stream_duration = _safe_float(primary.duration)
        if stream_duration is not None:
            stream_duration *= _safe_float(primary.time_base) or 0.0
        bit_rate = primary.bit_rate or ctx.bit_rate
        audio = {
            "codec": codec,
            "profile": profile,
            "channels": channels,
            "layout": layout,
            "sample_rate_hz": int(ctx.sample_rate or 0) or None,
            "bit_rate": int(bit_rate) if bit_rate else None,
            "stream_index": primary.index,
            "audio_streams": len(streams),
            "duration_s": stream_duration or duration_s,
        }

    is_atmos = "atmos" in (audio["profile"] or "").lower()
    if not is_atmos:
        # Older libav builds don't decorate the profile string. Fall
        # back to the layout heuristic, but keep it labelled as such
        # so the UI can say "likely" instead of promising.
        is_atmos = audio["codec"] in _ATMOS_CODECS and audio["channels"] > 2
        confidence = "heuristic" if is_atmos else "none"
    else:
        confidence = "container"

    audio["is_atmos"] = bool(is_atmos)
    audio["atmos_confidence"] = confidence

    notes: list[str] = []
    if confidence == "container":
        notes.append(
            "Container profile reports "
            f"'{audio['profile']}' — the E-AC-3 JOC object layer is present."
        )
    elif confidence == "heuristic":
        notes.append(
            f"{audio['codec'].upper()} with {audio['channels']} channels, "
            "but the container does not name an Atmos profile."
        )
    if audio["channels"] > 2:
        notes.append(
            "PyAV (the Tideway engine) renders the "
            f"{audio['layout'] or str(audio['channels']) + 'ch'} core bed "
            "only; height/object rendering happens in the Windows pipeline."
        )
    else:
        notes.append(
            "Stereo source — there is no surround bed for either engine "
            "to render."
        )

    return {
        "path": path,
        "exists": True,
        "size_bytes": size_bytes,
        "container": container_name,
        "duration_s": duration_s,
        "audio": audio,
        "notes": notes,
    }

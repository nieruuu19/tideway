"""Dolby Atmos (E-AC-3 JOC) stream resolution for TIDAL.

TIDAL stores Atmos mixes as their own catalog entries and serves them
through the OpenAPI v2 `trackManifests` endpoint with a `formats=EAC3_JOC`
filter. That endpoint returns a base64 DASH (MPD) data URI, which this
module expands into the `[init, segment, ...]` URL list that
`SegmentReader` already knows how to concatenate into one fragmented
MP4 for PyAV.

The standard v1 playback endpoint (`Track.get_stream`) never returns the
Atmos asset — it hands back the stereo downmix at the requested quality —
so an Atmos play *must* go through this module.

What comes out of the E-AC-3 decoder is the 5.1 (or 7.1) core bed. The
JOC object metadata rides alongside the bed and is not rendered by
FFmpeg; object-aware rendering needs Atmos hardware / a licensed
renderer. Playing the core bed is the same thing every non-Atmos player
does with these assets, and it is far better than the stereo fold-down.
"""
from __future__ import annotations

import base64
import logging
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin

# Tidal manifests are small (~10-40 KB) but the parser is fed data from
# the network, so keep the same hard caps tidekeeper uses: a malformed
# or hostile manifest must fail fast instead of expanding into millions
# of URLs.
try:
    from defusedxml import ElementTree as ET
except Exception:  # pragma: no cover - defusedxml ships with tidalapi
    from xml.etree import ElementTree as ET  # type: ignore[no-redef]

log = logging.getLogger(__name__)

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_NODES = 200_000
MAX_SEGMENTS = 100_000
MAX_EXPANDED_URL_BYTES = 16 * 1024 * 1024
MAX_TEMPLATE_WIDTH = 20

ATMOS_FORMAT = "EAC3_JOC"
ATMOS_MODE = "DOLBY_ATMOS"

# Cap the per-process catalog caches. A browse-heavy session can touch
# thousands of tracks; negative results matter too (they stop us from
# re-walking an artist's discography for every track on a non-Atmos
# album), so both hits and misses are cached.
_CACHE_MAX = 512
_CACHE_LOCK = threading.Lock()
_TRACK_TWIN_CACHE: "OrderedDict[str, Optional[int]]" = OrderedDict()
_ALBUM_TWIN_CACHE: "OrderedDict[str, Optional[int]]" = OrderedDict()


class AtmosUnavailable(RuntimeError):
    """TIDAL has no playable E-AC-3 JOC asset for this track.

    Raised both for "this release has no Atmos mix" and for malformed /
    DRM-protected manifests. The player treats it as a quality fallback
    signal, not a fatal error.
    """


@dataclass(frozen=True)
class AtmosStream:
    urls: list[str]
    codec: str = "ec-3"
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


# ---------------------------------------------------------------------------
# DASH parsing
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        if len(value) > MAX_MANIFEST_BYTES:
            raise AtmosUnavailable("Dolby Atmos manifest is too large.")
        return value.decode("utf-8-sig")
    value = str(value)
    if len(value.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise AtmosUnavailable("Dolby Atmos manifest is too large.")
    return value


def _seconds(value: Optional[str]) -> Optional[float]:
    match = re.fullmatch(
        r"PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?",
        value or "",
    )
    if not match:
        return None
    return sum(
        float(part or 0) * factor
        for part, factor in zip(match.groups(), (3600, 60, 1))
    )


def _integer(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expand_template(
    pattern: str, representation: Any, number: int, timestamp: int
) -> str:
    values = {
        "RepresentationID": representation.get("id", ""),
        "Bandwidth": representation.get("bandwidth", ""),
        "Number": number,
        "Time": timestamp,
    }

    def replace(match: "re.Match[str]") -> str:
        value = values[match.group(1)]
        width = int(match.group(2) or 0)
        if width > MAX_TEMPLATE_WIDTH:
            raise AtmosUnavailable("DASH number formatting width is too large.")
        return str(value).zfill(width)

    # `$$` is an escaped literal dollar in the DASH spec. Swap it aside
    # before the substitution pass so it isn't matched as a token.
    result = re.sub(
        r"\$(RepresentationID|Bandwidth|Number|Time)(?:%0(\d+)d)?\$",
        replace,
        pattern.replace("$$", "\x00"),
    ).replace("\x00", "$")
    if not result:
        raise AtmosUnavailable("DASH segment URL is empty.")
    return result


def parse_dash_audio(xml_text: Any) -> list[dict]:
    """Expand an MPD into per-representation URL lists and quality facts.

    Only audio representations are returned. Each entry is a dict with
    `urls` (init segment first), `codec`, `mimeType`, `bandwidth`,
    `sampleRate`, `bitDepth`, `channels`, `id`.

    Namespace-agnostic: Tidal's MPDs use the DASH namespace, but some
    CDN variants have shipped without it, so tags are compared after
    stripping any `{namespace}` prefix.
    """
    try:
        root = ET.fromstring(_text(xml_text))
    except AtmosUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed XML of any kind
        raise AtmosUnavailable("Dolby Atmos manifest is invalid XML.") from exc

    for element in root.iter():
        element.tag = str(element.tag).rsplit("}", 1)[-1]
        if element.tag == "ContentProtection":
            # Which variant TIDAL serves is decided by the OAuth client
            # that minted the token; the mobile (PKCE) client gets a
            # Widevine-wrapped stream. The remedy is a different login
            # client, not a parse fallback, so say so.
            raise AtmosUnavailable(
                "TIDAL returned a DRM-protected Dolby Atmos stream for "
                "this login. Sign out, choose another TIDAL client on "
                "the login screen, and try again."
            )

    if sum(1 for _ in root.iter()) > MAX_MANIFEST_NODES:
        raise AtmosUnavailable("DASH manifest contains too many XML nodes.")

    def base_url(parent: str, node: Any) -> str:
        child = node.find("BaseURL")
        if child is not None and child.text:
            return urljoin(parent, child.text.strip())
        return parent

    periods = root.findall("Period")
    if len(periods) != 1:
        raise AtmosUnavailable(
            "DASH requires a single complete period; multi-period audio "
            "is not supported."
        )

    tracks: list[dict] = []
    expanded_count = 0
    expanded_bytes = 0

    for period in periods:
        duration = _seconds(period.get("duration"))
        if duration is None:
            duration = _seconds(root.get("mediaPresentationDuration"))
            if duration is not None:
                duration -= _seconds(period.get("start")) or 0

        for adaptation in period.findall("AdaptationSet"):
            for rep in adaptation.findall("Representation"):
                media_type = rep.get("mimeType", adaptation.get("mimeType", ""))
                if (
                    adaptation.get("contentType") != "audio"
                    and not media_type.startswith("audio/")
                ):
                    continue

                # SegmentTemplate attributes cascade Period ->
                # AdaptationSet -> Representation; the innermost wins.
                # SegmentTimeline lives inside whichever node defines it.
                attrs: dict[str, str] = {}
                timeline = None
                for node in (period, adaptation, rep):
                    template = node.find("SegmentTemplate")
                    if template is not None:
                        attrs.update(template.attrib)
                        found = template.find("SegmentTimeline")
                        if found is not None:
                            timeline = found

                if not attrs.get("initialization") or not attrs.get("media"):
                    raise AtmosUnavailable(
                        "DASH requires initialization and media segment templates."
                    )

                prefix = base_url(
                    base_url(base_url(base_url("", root), period), adaptation),
                    rep,
                )
                number = int(attrs.get("startNumber", 1))
                scale = int(attrs.get("timescale", 1) or 1)
                if scale <= 0:
                    raise AtmosUnavailable("Invalid DASH timescale.")

                timestamps: list[int] = []
                if timeline is not None:
                    current = 0
                    entries = list(timeline.findall("S"))
                    for index, entry in enumerate(entries):
                        current = int(entry.get("t", current))
                        step = int(entry.get("d", 0))
                        if step <= 0:
                            raise AtmosUnavailable("Invalid DASH segment duration.")
                        repeat = int(entry.get("r", 0))
                        if repeat < 0:
                            # Negative r means "repeat until the next S.t"
                            # (or, for the last entry, until the period
                            # end). Unbounded and unresolvable timelines
                            # are rejected rather than looped forever.
                            following = (
                                entries[index + 1]
                                if index + 1 < len(entries)
                                else None
                            )
                            end = (
                                int(following.get("t"))
                                if following is not None and following.get("t")
                                else None
                            )
                            if end is None and duration is not None:
                                end = duration * scale + int(
                                    attrs.get("presentationTimeOffset", 0)
                                )
                            if end is None:
                                raise AtmosUnavailable(
                                    "Unbounded DASH timeline is not supported."
                                )
                            repeat = math.ceil((end - current) / step) - 1
                        count = repeat + 1
                        if count < 0 or len(timestamps) + count > MAX_SEGMENTS:
                            raise AtmosUnavailable(
                                "Invalid or excessively large DASH timeline."
                            )
                        timestamps.extend(
                            current + offset * step for offset in range(count)
                        )
                        current += step * count
                elif duration is not None and int(attrs.get("duration", 0) or 0) > 0:
                    step = int(attrs["duration"])
                    count = math.ceil(duration * scale / step)
                    if count > MAX_SEGMENTS:
                        raise AtmosUnavailable("DASH timeline is too large.")
                    timestamps = [offset * step for offset in range(count)]
                else:
                    raise AtmosUnavailable(
                        "DASH manifest has no finite segment timeline."
                    )

                if (
                    not timestamps
                    or expanded_count + len(timestamps) + 1 > MAX_SEGMENTS
                ):
                    raise AtmosUnavailable(
                        "DASH manifest has no media segments or too many segments."
                    )

                urls: list[str] = []
                for pattern, segment_number, timestamp in [
                    (attrs["initialization"], number, 0),
                    *(
                        (attrs["media"], number + index, timestamp)
                        for index, timestamp in enumerate(timestamps)
                    ),
                ]:
                    location = urljoin(
                        prefix,
                        _expand_template(pattern, rep, segment_number, timestamp),
                    )
                    expanded_bytes += len(location.encode("utf-8"))
                    if expanded_bytes > MAX_EXPANDED_URL_BYTES:
                        raise AtmosUnavailable("Expanded DASH manifest is too large.")
                    urls.append(location)
                expanded_count += len(urls)

                representation_id = rep.get("id", "")
                bit_depth = rep.get(
                    "audioBitDepth", adaptation.get("audioBitDepth", "")
                )
                if not bit_depth:
                    match = re.fullmatch(
                        r"FLAC(?:_[A-Z0-9]+)*,\d+,(\d{1,2})",
                        representation_id,
                        re.IGNORECASE,
                    )
                    bit_depth = match.group(1) if match else ""

                channel_node = rep.find("AudioChannelConfiguration")
                if channel_node is None:
                    channel_node = adaptation.find("AudioChannelConfiguration")
                channels = (
                    channel_node.get("value", "") if channel_node is not None else ""
                )

                tracks.append(
                    {
                        "urls": urls,
                        "id": representation_id,
                        "codec": rep.get("codecs", adaptation.get("codecs", "")),
                        "mimeType": media_type,
                        "bandwidth": _integer(rep.get("bandwidth")),
                        "sampleRate": _integer(
                            rep.get(
                                "audioSamplingRate",
                                adaptation.get("audioSamplingRate", ""),
                            )
                        ),
                        "bitDepth": _integer(bit_depth),
                        "channels": _integer(channels),
                    }
                )

    if not tracks:
        raise AtmosUnavailable(
            "DASH manifest contains no supported audio representations."
        )
    return tracks


def _is_eac3(representation: dict) -> bool:
    codec = str(representation.get("codec") or "").lower()
    return "ec-3" in codec or "eac3" in codec


def _quality_key(representation: dict) -> tuple:
    return (
        int(_is_eac3(representation)),
        int(representation.get("bitDepth") or 0),
        int(representation.get("sampleRate") or 0),
        int(representation.get("bandwidth") or 0),
        int(representation.get("channels") or 0),
    )


def best_dash_representation(xml_text: Any) -> dict:
    """Pick the highest-fidelity audio representation.

    E-AC-3 sorts above everything else: an Atmos manifest that also
    carries a stereo AAC fallback must resolve to the Atmos asset, not
    to whichever representation XML happens to list first.
    """
    return max(parse_dash_audio(xml_text), key=_quality_key)


# ---------------------------------------------------------------------------
# OpenAPI manifest fetch
# ---------------------------------------------------------------------------


def _decode_manifest_data_uri(uri: Any) -> str:
    if not isinstance(uri, str) or "," not in uri:
        raise AtmosUnavailable("Dolby Atmos manifest is empty.")
    header, payload = uri.split(",", 1)
    if ";base64" not in header.lower():
        raise AtmosUnavailable("Dolby Atmos manifest encoding is unsupported.")
    if len(payload) > 4 * ((MAX_MANIFEST_BYTES + 2) // 3):
        raise AtmosUnavailable("Dolby Atmos manifest is too large.")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise AtmosUnavailable("Dolby Atmos manifest is invalid.") from exc
    if len(decoded) > MAX_MANIFEST_BYTES:
        raise AtmosUnavailable("Dolby Atmos manifest is too large.")
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AtmosUnavailable("Dolby Atmos manifest is invalid.") from exc


def _is_unavailable_error(exc: BaseException) -> bool:
    """Classify an HTTP failure as "no Atmos asset" vs. "something broke".

    403/404/405 are how Tidal says "this client isn't entitled to / this
    release doesn't have this format". 401 and 429 must propagate: the
    caller refreshes the token on the former and engages the request
    gate's backoff on the latter.
    """
    if type(exc).__name__ in (
        "ObjectNotFound",
        "AssetNotAvailable",
        "StreamNotAvailable",
    ):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in (400, 403, 404, 405)


def _openapi_base(session: Any) -> str:
    config = getattr(session, "config", None)
    return (
        getattr(config, "openapi_v2_location", None)
        or "https://openapi.tidal.com/v2/"
    )


def fetch_manifest_attributes(
    session: Any,
    track_id: int,
    usage: str,
    *,
    formats: str = ATMOS_FORMAT,
    label: str = "Dolby Atmos",
) -> dict:
    params = {
        "manifestType": "MPEG_DASH",
        "uriScheme": "DATA",
        "usage": usage,
        "adaptive": "false",
        "formats": formats,
    }
    headers = {"Accept": "application/vnd.api+json"}
    try:
        response = session.request.request(
            "GET",
            f"trackManifests/{int(track_id)}",
            params=params,
            headers=headers,
            base_url=_openapi_base(session),
        )
    except Exception as exc:
        if _is_unavailable_error(exc):
            raise AtmosUnavailable(
                f"{label} manifest request failed: {exc}"
            ) from exc
        raise

    try:
        payload = response.json()
    except ValueError as exc:
        raise AtmosUnavailable(
            f"{label} manifest request returned invalid JSON."
        ) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    attributes = data.get("attributes") if isinstance(data, dict) else None
    if not isinstance(attributes, dict):
        raise AtmosUnavailable(
            f"{label} manifest response is missing its attributes."
        )

    presentation = (
        str(attributes.get("trackPresentation") or "FULL").strip().upper()
    )
    if presentation == "PREVIEW":
        raise AtmosUnavailable(
            f"TIDAL returned a preview-only {label} stream."
        )
    return attributes


def normalization_facts(
    attributes: dict, scope: str
) -> tuple[Optional[float], Optional[float]]:
    data = attributes.get(f"{scope}AudioNormalizationData")
    if not isinstance(data, dict):
        return None, None

    def number(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return number(data.get("replayGain")), number(data.get("peakAmplitude"))


def fetch_atmos_stream(session: Any, track_id: int) -> AtmosStream:
    """Fetch and expand the Atmos manifest for one catalog track id."""
    attributes: Optional[dict] = None
    last_error: Optional[AtmosUnavailable] = None
    # PLAYBACK first: it's the usage a player is entitled to on every
    # tier. DOWNLOAD is a fallback for accounts/regions where the
    # playback manifest is refused but the asset itself is available —
    # the bytes are identical.
    for usage in ("PLAYBACK", "DOWNLOAD"):
        try:
            attributes = fetch_manifest_attributes(session, track_id, usage)
            break
        except AtmosUnavailable as exc:
            last_error = exc
            log.debug(
                "Atmos manifest usage=%s unavailable for track %s: %s",
                usage,
                track_id,
                exc,
            )
    if attributes is None:
        raise last_error or AtmosUnavailable(
            "Dolby Atmos stream is not available for this track."
        )

    formats = attributes.get("formats") or []
    if ATMOS_FORMAT not in formats:
        raise AtmosUnavailable(
            "Dolby Atmos stream is not available for this track."
        )

    xml_data = _decode_manifest_data_uri(attributes.get("uri") or "")
    representation = best_dash_representation(xml_data)
    if not _is_eac3(representation):
        raise AtmosUnavailable(
            "Dolby Atmos manifest did not contain an E-AC-3 representation."
        )

    track_gain, track_peak = normalization_facts(attributes, "track")
    album_gain, album_peak = normalization_facts(attributes, "album")
    return AtmosStream(
        urls=list(representation["urls"]),
        codec=representation.get("codec") or "ec-3",
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


# ---------------------------------------------------------------------------
# Catalog twin lookup
# ---------------------------------------------------------------------------


def is_atmos(obj: Any) -> bool:
    modes = getattr(obj, "audio_modes", None) or []
    return any(str(mode).strip().upper() == ATMOS_MODE for mode in modes)


def _normalize_title(title: Any) -> str:
    text = " ".join(str(title or "").strip().lower().split())
    # Some Atmos editions append the mode to the album title
    # ("Random Access Memories (Dolby Atmos)"). Strip it so the
    # stereo/Atmos pair still compares equal.
    text = re.sub(r"\s*[\(\[]\s*dolby\s*atmos\s*[\)\]]\s*$", "", text)
    text = re.sub(r"\s*-\s*dolby\s*atmos\s*$", "", text)
    return text.strip()


def _cache_get(cache: "OrderedDict[str, Optional[int]]", key: str) -> Optional[int]:
    with _CACHE_LOCK:
        if key in cache:
            return cache[key]
    return None


def _cache_has(cache: "OrderedDict[str, Optional[int]]", key: str) -> bool:
    with _CACHE_LOCK:
        return key in cache


def _cache_put(
    cache: "OrderedDict[str, Optional[int]]", key: str, value: Optional[int]
) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)


def _artist_id(album: Any) -> Optional[int]:
    artist = getattr(album, "artist", None)
    artist_id = _integer(getattr(artist, "id", None)) if artist is not None else None
    if artist_id is not None:
        return artist_id
    for candidate in getattr(album, "artists", None) or []:
        artist_id = _integer(getattr(candidate, "id", None))
        if artist_id is not None:
            return artist_id
    return None


def _artist_albums(session: Any, artist_id: int) -> list:
    try:
        artist = session.artist(artist_id)
    except Exception as exc:  # noqa: BLE001 - twin lookup is best-effort
        log.info("Atmos twin lookup: artist %s unavailable: %s", artist_id, exc)
        return []
    albums: list = []
    try:
        albums.extend(list(artist.get_albums(limit=100) or []))
    except Exception as exc:  # noqa: BLE001
        log.info("Atmos twin lookup: album list for %s failed: %s", artist_id, exc)
    try:
        albums.extend(list(artist.get_ep_singles(limit=100) or []))
    except Exception as exc:  # noqa: BLE001
        log.debug("Atmos twin lookup: EP/single list for %s failed: %s", artist_id, exc)

    deduped: list = []
    seen: set[int] = set()
    for album in albums:
        album_id = _integer(getattr(album, "id", None))
        if album_id is None or album_id in seen:
            continue
        seen.add(album_id)
        deduped.append(album)
    return deduped


def _find_atmos_album(session: Any, album: Any) -> Optional[Any]:
    """Return the Atmos catalog twin of `album`, or None."""
    album_id = getattr(album, "id", None)
    album_key = str(album_id or "")
    if album_key and _cache_has(_ALBUM_TWIN_CACHE, album_key):
        cached = _cache_get(_ALBUM_TWIN_CACHE, album_key)
        if cached is None:
            return None
        try:
            return session.album(cached)
        except Exception:  # noqa: BLE001
            return None

    target_title = _normalize_title(getattr(album, "name", None))
    artist_id = _artist_id(album)
    if not target_title or artist_id is None:
        if album_key:
            _cache_put(_ALBUM_TWIN_CACHE, album_key, None)
        return None

    target_tracks = _integer(getattr(album, "num_tracks", None)) or 0
    target_explicit = bool(getattr(album, "explicit", False))

    matches = [
        candidate
        for candidate in _artist_albums(session, artist_id)
        if is_atmos(candidate)
        and _normalize_title(getattr(candidate, "name", None)) == target_title
    ]
    if not matches:
        if album_key:
            _cache_put(_ALBUM_TWIN_CACHE, album_key, None)
        return None

    def score(candidate: Any) -> tuple:
        candidate_tracks = _integer(getattr(candidate, "num_tracks", None)) or 0
        explicit_mismatch = int(
            bool(getattr(candidate, "explicit", False)) != target_explicit
        )
        return (
            abs(candidate_tracks - target_tracks),
            explicit_mismatch,
            str(getattr(candidate, "id", "")),
        )

    best = sorted(matches, key=score)[0]
    best_id = _integer(getattr(best, "id", None))
    if album_key:
        _cache_put(
            _ALBUM_TWIN_CACHE,
            album_key,
            None if best_id == _integer(album_id) else best_id,
        )
    if best_id == _integer(album_id):
        return None
    return best


def _match_twin_track(source: Any, candidates: list) -> Optional[Any]:
    target_isrc = str(getattr(source, "isrc", None) or "").strip().upper()
    if target_isrc:
        for candidate in candidates:
            candidate_isrc = str(getattr(candidate, "isrc", None) or "")
            if candidate_isrc.strip().upper() == target_isrc:
                return candidate

    target_title = _normalize_title(getattr(source, "title", None))
    target_number = _integer(getattr(source, "track_num", None))
    target_volume = _integer(getattr(source, "volume_num", None))

    if target_number is not None:
        for candidate in candidates:
            if _integer(getattr(candidate, "track_num", None)) != target_number:
                continue
            candidate_volume = _integer(getattr(candidate, "volume_num", None))
            if target_volume is not None and candidate_volume != target_volume:
                continue
            if _normalize_title(getattr(candidate, "title", None)) == target_title:
                return candidate

    for candidate in candidates:
        if _normalize_title(getattr(candidate, "title", None)) == target_title:
            return candidate
    return None


def find_atmos_track_id(session: Any, track: Any) -> Optional[int]:
    """Return the catalog id carrying this track's Atmos asset.

    TIDAL gives the Atmos mix its own track id, usually on its own album
    edition. When the track already advertises `DOLBY_ATMOS` in its
    audio modes, its own id is the answer. Otherwise we resolve the
    album's Atmos twin and match the track inside it by ISRC, then
    disc/track number + title, then title alone.
    """
    track_id = _integer(getattr(track, "id", None))
    if track_id is None:
        return None
    if is_atmos(track):
        return track_id

    track_key = str(track_id)
    if _cache_has(_TRACK_TWIN_CACHE, track_key):
        return _cache_get(_TRACK_TWIN_CACHE, track_key)

    result: Optional[int] = None
    album_ref = getattr(track, "album", None)
    album_id = _integer(getattr(album_ref, "id", None))
    if album_id is not None:
        try:
            album = session.album(album_id)
        except Exception as exc:  # noqa: BLE001 - best-effort lookup
            log.info(
                "Atmos twin lookup: album %s unavailable for track %s: %s",
                album_id,
                track_id,
                exc,
            )
            album = None
        if album is not None:
            if is_atmos(album):
                # The parent album is the Atmos edition, so the track's
                # own id is the asset even though its audioModes list
                # didn't advertise the mode.
                result = track_id
            else:
                twin_album = _find_atmos_album(session, album)
                if twin_album is not None:
                    try:
                        candidates = list(twin_album.tracks() or [])
                    except Exception as exc:  # noqa: BLE001
                        log.info(
                            "Atmos twin lookup: track list for album %s failed: %s",
                            getattr(twin_album, "id", None),
                            exc,
                        )
                        candidates = []
                    matched = _match_twin_track(track, candidates)
                    result = _integer(getattr(matched, "id", None))

    log.info(
        "Atmos twin lookup: track %s -> %s",
        track_id,
        result if result is not None else "none",
    )
    _cache_put(_TRACK_TWIN_CACHE, track_key, result)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_atmos_playback(session: Any, track_id: int) -> tuple[Any, AtmosStream]:
    """Resolve `track_id` to (metadata track, expanded Atmos stream).

    The metadata `Track` is the one the user asked for — the Atmos twin
    shares its title/artist anyway, and using the original keeps the UI
    and TIDAL play reporting anchored to the release the user clicked.

    Metadata is *optional*: some ids (v2-only catalog entries, removed
    or region-shifted releases) 404 in the v1 catalog while their
    playback manifest is still served. A metadata failure therefore
    skips the twin lookup and tries the requested id directly, instead
    of vetoing a track that will play. Returns `None` as the track in
    that case.
    """
    try:
        track = session.track(track_id)
    except Exception as exc:  # noqa: BLE001 - metadata must not veto playback
        log.info(
            "Atmos resolve: v1 metadata unavailable for %s (%s); "
            "using the manifest service directly",
            track_id,
            exc,
        )
        track = None

    ids: list[int] = []
    twin_id = find_atmos_track_id(session, track) if track is not None else None
    if twin_id is not None:
        ids.append(twin_id)
    if track_id not in ids:
        # Direct id as a last resort: catalog metadata occasionally lags
        # behind entitlements, and the manifest service is authoritative.
        ids.append(track_id)

    last_error: Optional[AtmosUnavailable] = None
    for candidate in ids:
        try:
            stream = fetch_atmos_stream(session, candidate)
        except AtmosUnavailable as exc:
            last_error = exc
            log.info("Atmos manifest for candidate %s unavailable: %s", candidate, exc)
            continue
        return track, stream
    raise last_error or AtmosUnavailable(
        "Dolby Atmos stream is not available for this track."
    )

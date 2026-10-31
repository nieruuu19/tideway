import json
import logging
import os
import re
import stat
import sys
import tempfile
import threading
import time
import webbrowser
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Tuple

import tidalapi

from app import api_keys
from app import lossless as lossless_mod
from app.http import network_error_classes
from app.paths import user_data_dir
from app.settings import load_settings, save_settings

# audio.log is bound to the "tideway.audio" logger in
# app/audio/player.py (a RotatingFileHandler, propagate=False).
# Auth-lifecycle lines used to print only to stderr, which is
# invisible in the packaged app — so a user "getting logged out"
# left no diagnosable trace in production. Mirror the high-signal
# auth lines to audio.log as well, same pattern the AirPlay
# diagnostics use.
_audit_log = logging.getLogger("tideway.audio")


def _tlog(msg: str) -> None:
    """High-signal Tidal auth line: dev console (stderr) + audio.log."""
    line = f"[tidal] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        _audit_log.info(line)
    except Exception:
        # Logging to file must never break the auth path; the
        # stderr line already carried the signal.
        pass


def _now_matching(expiry: datetime) -> datetime:
    """The current time in the same frame as `expiry`.

    Token expiry is stored as *naive UTC*: tidalapi builds it with
    `datetime.utcnow()` (session.py) and `_token_refresh_capturing`
    matches that convention deliberately. Comparing such a value
    against `datetime.now()` compares UTC against local time, so the
    remaining lifetime comes out wrong by the machine's UTC offset in
    whichever direction that offset points:

      - West of UTC the remaining time is *overstated*. The refresh
        watchdog only acts inside a 5 minute window, so an offset of
        several hours means it never fires at all: the token dies
        while the countdown still reads hours, and `load_session`'s
        `expiry <= now` test misses for the same reason, so the launch
        refresh is skipped too. The user is silently signed out with
        no log line explaining it.
      - East of UTC it is *understated*, so refreshes fire an offset
        early and every launch takes the refresh path, rotating the
        refresh token far more often than necessary.

    Only machines sitting exactly on UTC behave correctly, which is
    why this survived: the tests built their fixture expiry with
    `datetime.now()` too, so they agreed with the reader and never
    exercised the frame the writer actually uses (#309).

    An aware `expiry` is handled for its own sake, so this stays
    correct if tidalapi ever moves off naive datetimes.
    """
    tz = getattr(expiry, "tzinfo", None)
    if tz is not None:
        return datetime.now(tz)
    return datetime.now(timezone.utc).replace(tzinfo=None)


SESSION_FILE = user_data_dir() / "tidal_session.json"


def _stored_refresh_token() -> Optional[str]:
    """The refresh token currently on disk, if any.

    Read at save time so an in-memory session that has lost its refresh
    token cannot overwrite a good one. Returns None when the file is
    missing or unreadable, which puts the caller back where it already
    was rather than inventing a credential.
    """
    try:
        with open(SESSION_FILE) as f:
            return json.load(f).get("refresh_token") or None
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # Unreadable or malformed. Nothing to preserve, and the caller
        # is about to replace the file anyway.
        return None

# Connection-level failures from either HTTP transport. Loading a
# persisted session over a dead network has to be told apart from
# Tidal rejecting the credentials — see load_session.
_NETWORK_ERRORS = network_error_classes()


# --- Tidal request gate + abuse/rate-limit backoff ---------------------
#
# Tidal's anti-abuse layer responds to suspicious volume with either
# HTTP 429 (soft rate-limit) or HTTP 403 + `"abuse_detected"` (account
# suspension escalation path). When we see either signal we want to
# stop hitting Tidal immediately: further requests only deepen the
# suspension and can trigger permanent-ban. We install one interceptor
# on tidalapi's central HTTP method so every call passes through the
# gate. Module-level state so any place that instantiates a
# TidalClient shares the same backoff window.
_tidal_backoff_until: float = 0.0
_tidal_backoff_reason: str = ""
_tidal_backoff_lock = threading.Lock()


class TidalBackoffError(Exception):
    """Raised when we refuse to make a Tidal request because we're
    still inside a backoff window. Contains the remaining seconds so
    the API layer can surface a 503 with a Retry-After-style message."""

    def __init__(self, seconds_remaining: float, reason: str):
        self.seconds_remaining = seconds_remaining
        self.reason = reason
        super().__init__(
            f"Tidal backoff active for another {seconds_remaining:.0f}s: {reason}"
        )


class TransientRefreshError(Exception):
    """Raised when a token refresh fails for a transient reason — a 429
    rate-limit, a 5xx from Tidal, or a network error — rather than a
    genuinely dead refresh token.

    The distinction matters because force_refresh logs the user out and
    deletes the session file on a *permanent* auth failure
    (AuthenticationError). A transient failure must NOT do that: the
    refresh token is still valid, so we keep the session and let the
    watchdog retry on its next tick. Conflating the two was silently
    logging users out days later whenever a background refresh happened
    to land during a rate-limit window or a Tidal outage."""


def tidal_backoff_state() -> dict:
    """Snapshot for /api/tidal/backoff so the frontend can surface a
    banner explaining why things are blocked."""
    with _tidal_backoff_lock:
        return {
            "active": time.time() < _tidal_backoff_until,
            "until_epoch": _tidal_backoff_until,
            "seconds_remaining": max(0.0, _tidal_backoff_until - time.time()),
            "reason": _tidal_backoff_reason,
        }


def _classify_tidal_error(status_code: Optional[int], body: str) -> None:
    """Inspect a Tidal response's status + body and engage the
    appropriate backoff. No-op on success / non-abuse failures.
    Shared between the session request gate (for normal API calls)
    and the PKCE token-endpoint exception handler (which bypasses
    the gate because tidalapi uses its own requests.post for auth).
    """
    if status_code == 429:
        _trigger_tidal_backoff(60.0, "rate-limited (HTTP 429)")
        return
    if status_code == 403:
        body_lower = (body or "").lower()
        if "abuse" in body_lower or "suspended" in body_lower:
            _trigger_tidal_backoff(
                30 * 60.0, f"abuse-detected (HTTP 403): {(body or '')[:200]}"
            )


def _trigger_tidal_backoff(duration_seconds: float, reason: str) -> None:
    global _tidal_backoff_until, _tidal_backoff_reason
    with _tidal_backoff_lock:
        until = time.time() + duration_seconds
        # Extend if we'd be setting a shorter window than already
        # pending — never pull a backoff in early.
        if until > _tidal_backoff_until:
            _tidal_backoff_until = until
            _tidal_backoff_reason = reason
    import sys as _sys
    print(
        f"[tidal] backoff engaged for {duration_seconds:.0f}s: {reason}",
        file=_sys.stderr,
        flush=True,
    )


def tidal_jitter_sleep() -> None:
    """Random 50-200 ms pause used inside parallel pools that hit
    Tidal. Aligns bursts to look closer to human-paced navigation
    than a mechanical fan-out — our five concurrent requests all
    firing within 5 ms is a signature no real client produces."""
    import random as _random
    time.sleep(_random.uniform(0.05, 0.20))


# Headers we apply on top of whatever curl-cffi's impersonation
# profile already supplies. UA matches the real Tidal Android client;
# Accept-Language pins us to en-US so personalized editorial doesn't
# vary with the (often empty) system locale.
_TIDAL_ANDROID_HEADERS: dict = {
    "User-Agent": "TIDAL_ANDROID/2.88.0 okhttp/4.12.0",
    "Accept-Language": "en-US",
}


def _install_tidal_request_gate(session: tidalapi.Session) -> None:
    """Wrap tidalapi's Requests.basic_request so every HTTP call:
      1. Refuses early with TidalBackoffError if we're still in a
         backoff window (preserves the account — every request while
         suspended extends the strike window).
      2. On 429, engages a 60 s backoff.
      3. On 403 with `abuse_detected` / `"abuse"` in the body, engages
         a much longer (30-minute) backoff so we don't re-hit while
         Tidal's fraud system is still watching.
    """
    requests_obj = session.request
    original = requests_obj.basic_request

    def wrapped(method, path, params=None, data=None, headers=None, base_url=None):
        now = time.time()
        with _tidal_backoff_lock:
            remaining = _tidal_backoff_until - now
            reason = _tidal_backoff_reason
        if remaining > 0:
            raise TidalBackoffError(remaining, reason)

        resp = original(method, path, params=params, data=data, headers=headers, base_url=base_url)
        body = ""
        if getattr(resp, "status_code", None) == 403:
            try:
                body = resp.text[:400]
            except Exception:
                pass
        _classify_tidal_error(getattr(resp, "status_code", None), body)
        return resp

    # Bind the wrapper on the instance so we don't leak patches
    # across Session objects (the login flow rebuilds the session).
    requests_obj.basic_request = wrapped


def _swap_to_impersonated_transport(session: tidalapi.Session) -> None:
    """Replace tidalapi's underlying request_session with a curl-cffi
    session that matches a real mobile-Chrome TLS stack. urllib3's
    ClientHello + HTTP/2 SETTINGS are a Python-specific fingerprint
    Tidal's anti-abuse can match on without seeing any request
    volume. Best-effort: keeps the plain requests.Session if
    curl-cffi can't be loaded.
    """
    from app.http import build_impersonated_session

    impersonated = build_impersonated_session()
    if impersonated is None:
        return
    old = getattr(session, "request_session", None)
    if old is not None and hasattr(old, "headers"):
        try:
            impersonated.headers.update(dict(old.headers))
        except Exception:
            pass
    impersonated.headers["User-Agent"] = _TIDAL_ANDROID_HEADERS["User-Agent"]
    impersonated.headers["Accept-Language"] = _TIDAL_ANDROID_HEADERS[
        "Accept-Language"
    ]
    session.request_session = impersonated


_track_ai_patched = False


def _patch_track_ai_field() -> None:
    """Teach tidalapi's Track & Album parsers to keep the `ai` flag,
    Dolby Atmos metadata (`audioModes` / `mediaMetadata.tags`), and
    Tidal's per-track availability flags (`streamReady` /
    `allowStreaming`) so listings can hide what Tidal won't play.
    """
    global _track_ai_patched
    if _track_ai_patched:
        return
    from tidalapi.media import Track, Album

    original_track = Track.parse_track
    Track.ai = None
    Track.audio_modes = []
    Track.media_metadata_tags = []
    Track.is_dolby_atmos = False
    Track.stream_ready = None
    Track.allow_streaming = None

    def parse_track_with_ai_and_atmos(self, json_obj, album=None):
        track = original_track(self, json_obj, album)
        ai = bool(json_obj.get("ai")) if "ai" in json_obj else None
        audio_modes = list(json_obj.get("audioModes") or [])
        meta_tags = list((json_obj.get("mediaMetadata") or {}).get("tags") or [])
        is_atmos = (
            "DOLBY_ATMOS" in audio_modes
            or "DOLBY_ATMOS" in meta_tags
            or getattr(self, "audio_mode", None) == "DOLBY_ATMOS"
        )
        stream_ready = json_obj.get("streamReady")
        allow_streaming = json_obj.get("allowStreaming")
        for target in (track, self):
            target.ai = ai
            target.audio_modes = audio_modes
            target.media_metadata_tags = meta_tags
            target.is_dolby_atmos = is_atmos
            target.stream_ready = stream_ready
            target.allow_streaming = allow_streaming
        return track

    Track.parse_track = parse_track_with_ai_and_atmos

    if hasattr(Album, "parse_album"):
        original_album = Album.parse_album
        Album.audio_modes = []
        Album.media_metadata_tags = []
        Album.is_dolby_atmos = False

        def parse_album_with_atmos(self, json_obj, artist=None, artists=None):
            alb = original_album(self, json_obj, artist, artists)
            audio_modes = list(json_obj.get("audioModes") or [])
            meta_tags = list((json_obj.get("mediaMetadata") or {}).get("tags") or [])
            is_atmos = "DOLBY_ATMOS" in audio_modes or "DOLBY_ATMOS" in meta_tags
            for target in (alb, self):
                target.audio_modes = audio_modes
                target.media_metadata_tags = meta_tags
                target.is_dolby_atmos = is_atmos
            return alb

        Album.parse_album = parse_album_with_atmos

    _track_ai_patched = True


_patch_track_ai_field()


def _friendly_pkce_error(exc: BaseException) -> Optional[str]:
    """Translate the cryptic exceptions tidalapi can raise during the
    PKCE token exchange into something a non-developer can act on.
    Returns None when the exception isn't one we have a friendlier
    string for, so the caller can fall back to the raw message.

    The big one is `ConnectionError("Connection aborted!", PermissionError(13, ...))`,
    which on Windows almost always means an antivirus or firewall is
    blocking outbound TCP from the bundled Python. The user gets a
    401 with no idea what to fix; we want them to know it's a local
    security product, not a Tidal-side rejection.
    """
    text = repr(exc)
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    # The PermissionError is sometimes nested as the inner-args entry of
    # the requests ConnectionError rather than via __cause__, so also
    # walk .args looking for it. Same id-tracking dedupe so a wrapped
    # arg already on the cause chain doesn't trigger reprocessing.
    for item in list(chain):
        for a in getattr(item, "args", ()) or ():
            if isinstance(a, BaseException) and id(a) not in seen:
                seen.add(id(a))
                chain.append(a)

    for inner in chain:
        if isinstance(inner, PermissionError):
            return (
                "The login request was blocked by your computer before it "
                "reached Tidal. This is almost always antivirus or firewall "
                "software (Bitdefender, Sophos, McAfee, Norton, Windows "
                "Defender, etc) refusing to let Tideway open a network "
                "connection. Add Tideway (or Tideway.exe) to your security "
                "software's allow list and try again."
            )
    if "Connection aborted" in text and "Permission denied" in text:
        return (
            "The login request was blocked by your computer before it "
            "reached Tidal. This is almost always antivirus or firewall "
            "software refusing to let Tideway open a network connection. "
            "Add Tideway to your security software's allow list and try again."
        )
    return None


def _fetch_all_pages(method) -> list:
    """Exhaustively fetch every item from a paginated tidalapi list method.

    Strategy, in order:
      1. Ask for a huge single page with newest-first ordering.
      2. If that returns nothing or throws, ask for a huge single page with no
         ordering (some older tidalapi versions don't accept the kwargs).
      3. If still nothing, page through manually in chunks of 50.
      4. As a final fallback, call with no arguments.
    """
    # 1) Single-shot ordered fetch
    for kwargs in (
        {"limit": 10000, "order": "DATE", "order_direction": "DESC"},
        {"limit": 10000},
    ):
        try:
            items = list(method(**kwargs))
            if items:
                return items
        except Exception:
            pass

    # 2) Manual pagination in chunks of 50
    collected: list = []
    seen_ids: set = set()
    offset = 0
    page_size = 50
    while True:
        page: list = []
        for kwargs in (
            {"limit": page_size, "offset": offset},
            {"limit": page_size, "offset": offset, "order": "DATE", "order_direction": "DESC"},
        ):
            try:
                page = list(method(**kwargs))
                break
            except Exception:
                continue
        if not page:
            break
        # Duplicate-detection safety: if the endpoint misbehaves and keeps
        # returning the same page regardless of offset (happens on some
        # tidalapi versions / under transient 5xx retries), `len(page)
        # == page_size` would loop until the 20k cap with a completely
        # duplicated `collected` list. Break as soon as a page contributes
        # zero new items.
        added = 0
        for obj in page:
            key = getattr(obj, "id", None)
            if key is None:
                key = getattr(obj, "uuid", None)
            if key is None:
                # Item without a stable identifier — append unconditionally
                # but don't count it as "new" for the progress check.
                collected.append(obj)
                continue
            if key in seen_ids:
                continue
            seen_ids.add(key)
            collected.append(obj)
            added += 1
        if added == 0:
            break
        if len(page) < page_size:
            break
        offset += page_size
        if offset > 20000:  # hard safety cap
            break
    if collected:
        return collected

    # 3) Last resort
    try:
        return list(method())
    except Exception:
        return []


def _sort_by_date_added(items: list) -> list:
    """Sort items newest-first using whichever timestamp attribute tidalapi
    exposes. If no such attribute is found, return the list unchanged."""
    if not items:
        return items

    candidates = ("user_date_added", "date_added", "created", "added_at", "dateAdded")

    def key(obj):
        for attr in candidates:
            value = getattr(obj, attr, None)
            if value is None:
                continue
            if isinstance(value, datetime):
                return value.timestamp()
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except Exception:
                try:
                    return float(value)
                except Exception:
                    continue
        return None

    keyed = [(key(o), i, o) for i, o in enumerate(items)]
    if all(k is None for k, _, _ in keyed):
        return items  # no timestamp available; preserve server order
    # Items without a timestamp fall to the bottom
    keyed.sort(key=lambda t: (t[0] is None, -(t[0] or 0), t[1]))
    return [o for _, _, o in keyed]


# Tidal returns a `highestSoundQuality` code on the subscription endpoint.
# Map those codes to the tidalapi.Quality member names we use everywhere
# else in the app, and keep them in ascending order so "everything at or
# below X" is a simple slice.
_SUB_TTL_SEC = 300.0  # 5 minutes; subscriptions change rarely.

_SUB_QUALITY_ORDER = ["low_96k", "low_320k", "high_lossless", "hi_res_lossless"]
_SUB_QUALITY_MAP = {
    "LOW": "low_96k",
    "HIGH": "low_320k",
    "LOSSLESS": "high_lossless",
    "HI_RES_LOSSLESS": "hi_res_lossless",
    # Tidal's current single-tier subscription reports `HI_RES` as the
    # `highestSoundQuality`, which includes hi-res FLAC (Max). Whether a
    # session can actually stream it depends on the OAuth client that
    # minted the token, not on the login flow: the Tidal HiRes client
    # (api_keys index 5) serves HI_RES_LOSSLESS over
    # `playbackinfopostpaywall` even for device-code sessions, while the
    # legacy `urlpostpaywall` endpoint answers AAC for every tier. See
    # `request_stream_manifest()`.
    "HI_RES": "hi_res_lossless",
    "MASTER": "hi_res_lossless",
    "HIFI": "high_lossless",
    "HIFI_PLUS": "hi_res_lossless",
}

# What `Stream.audio_quality` says TIDAL actually served, mapped onto the
# internal tier names used everywhere else.
_DELIVERED_QUALITY_MAP = {
    "LOW": "low_96k",
    "HIGH": "low_320k",
    "LOSSLESS": "high_lossless",
    "HI_RES_LOSSLESS": "hi_res_lossless",
}
_LOSSY_TIERS = frozenset({"low_96k", "low_320k"})
_LOSSLESS_TIERS = frozenset({"high_lossless", "hi_res_lossless"})


def delivered_quality(stream) -> Optional[str]:
    """The tier TIDAL actually served, read off the Stream object.

    None means TIDAL didn't say; callers treat that as "assume asked".
    """
    raw = getattr(stream, "audio_quality", None)
    if raw is None:
        return None
    return _DELIVERED_QUALITY_MAP.get(str(getattr(raw, "value", raw)).upper())


def is_lossy_quality(tier: Optional[str]) -> bool:
    """True for the AAC tiers; None (unknown) reports False."""
    return tier in _LOSSY_TIERS


class TidalAssetUnavailable(RuntimeError):
    """TIDAL knows this track but is not serving it for playback.

    HTTP 401/403 carrying subStatus 4005 ("Asset is not ready for
    playback") — a per-track licensing/regional refusal, not a stale
    token. Refreshing the session cannot fix it, so it must not be
    routed through the auth-recovery path.
    """


def _is_auth_status(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) in (401, 403)


def _is_asset_unavailable(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) not in (401, 403):
        return False
    try:
        text = str(getattr(response, "text", None) or "")
    except Exception:
        text = ""
    text = text.lower()
    return "4005" in text or "not ready for playback" in text


def _flac_fallback(
    session, track, requested, served: Optional[str]
) -> Optional[Tuple[Any, Any, str]]:
    """Resolve the OpenAPI FLAC manifest when v1 handed back AAC.

    Returns a `(stream, manifest, delivered_tier)` triple shaped like
    the v1 pair (only the attributes callers read), or None when this
    track/login has no playable FLAC — the caller then keeps the honest
    AAC stream. Subscription-tier mapping is left alone: the tier this
    returns is the one the returned bytes actually are.
    """
    track_id = getattr(track, "id", None)
    if track_id is None:
        return None
    try:
        asset = lossless_mod.fetch_lossless_stream(
            session,
            int(track_id),
            hi_res=requested == tidalapi.Quality.hi_res_lossless,
        )
    except lossless_mod.LosslessUnavailable as exc:
        _tlog(
            f"track {track_id}: TIDAL served {served!r} for a lossless "
            f"request and has no playable FLAC ({exc})"
        )
        return None
    except Exception as exc:  # noqa: BLE001 - never lose the AAC stream
        # A broken/odd session (unit-test doubles), a 401 for a client
        # that isn't entitled to the OpenAPI route, or a transient 5xx:
        # the v1 stream already succeeded, so fall back to it rather
        # than failing the play.
        _tlog(
            f"track {track_id}: FLAC manifest lookup failed "
            f"({type(exc).__name__}: {exc}); keeping {served!r}"
        )
        return None

    tier = asset.tier
    stream = SimpleNamespace(
        audio_quality="HI_RES_LOSSLESS" if asset.is_hi_res else "LOSSLESS",
        bit_depth=asset.bit_depth,
        sample_rate=asset.sample_rate,
        audio_mode=None,
        track_replay_gain=asset.track_replay_gain,
        track_peak_amplitude=asset.track_peak,
        album_replay_gain=asset.album_replay_gain,
        album_peak_amplitude=asset.album_peak,
    )
    manifest = SimpleNamespace(
        urls=list(asset.urls),
        codecs=asset.codec,
        file_extension="flac",
        is_encrypted=False,
        get_codecs=lambda: asset.codec,
    )
    _tlog(
        f"track {track_id}: TIDAL served {served!r} for a lossless "
        f"request; playing {tier!r} FLAC from the OpenAPI manifest "
        f"({asset.bit_depth or '?'}-bit {asset.sample_rate or '?'} Hz, "
        f"{len(asset.urls) - 1} segments)"
    )
    return stream, manifest, tier


def request_stream_manifest(
    session, track
) -> Tuple[Any, Any, Optional[str]]:
    """Resolve `(stream, manifest, delivered_tier)` at the tier currently
    set on `session.config.quality`.

    `tracks/{id}/playbackinfopostpaywall` is the only endpoint that
    serves lossless to a device-code session — the legacy
    `urlpostpaywall` behind `Track.get_url()` answers AAC 320 for every
    tier, which is why stereo "Lossless"/"Max" downloads used to land as
    `.m4a` AAC files. Two TIDAL behaviours get smoothed over here:

    * `LOSSLESS` can come back as AAC (`HIGH`) on clients that don't
      serve the CD tier. A lossless request that renders lossy goes
      straight to the OpenAPI `trackManifests` route for FLAC (see
      `_flac_fallback`), which serves the same master — that route is
      what makes "Lossless"/"Max" actually play FLAC on clients whose v1
      tier negotiation answers AAC.
    * Only when that route has no playable FLAC is the v1 stream
      retried once at `HI_RES_LOSSLESS`, which some clients do serve
      even though the CD tier answers AAC.
    * `HI_RES_LOSSLESS` on a client that isn't entitled answers 401/403.
      Retried once at `LOSSLESS` so the stream lands instead of failing.

    Rate limits and auth errors that survive the fallback propagate —
    the caller's refresh / backoff path owns those.
    """
    requested = session.config.quality
    wants_lossless = requested in (
        tidalapi.Quality.high_lossless,
        tidalapi.Quality.hi_res_lossless,
    )
    hi_res_blocked = False
    try:
        stream = track.get_stream()
    except Exception as exc:
        if _is_asset_unavailable(exc):
            raise TidalAssetUnavailable(
                "TIDAL isn't serving this track for playback "
                "(asset not ready)."
            ) from exc
        if not (
            _is_auth_status(exc)
            and requested == tidalapi.Quality.hi_res_lossless
        ):
            raise
        hi_res_blocked = True
        print(
            f"[tidal] hi-res refused for track "
            f"{getattr(track, 'id', '?')}; falling back to lossless",
            file=sys.stderr,
            flush=True,
        )
        session.config.quality = tidalapi.Quality.high_lossless
        try:
            stream = track.get_stream()
        except Exception as fallback_exc:
            if _is_asset_unavailable(fallback_exc):
                raise TidalAssetUnavailable(
                    "TIDAL isn't serving this track for playback "
                    "(asset not ready)."
                ) from fallback_exc
            raise
        finally:
            session.config.quality = requested

    delivered = delivered_quality(stream)
    if wants_lossless and delivered in _LOSSY_TIERS:
        # Asked first, before another v1 round-trip: the OpenAPI route
        # is the one that reliably serves FLAC on clients whose v1 tier
        # negotiation answers AAC, and it costs a single request. The
        # hi-res v1 retry below costs a full request of its own and on
        # those same clients answers AAC again — so spending it first
        # doubled (and with the retry's own fallback, tripled) the API
        # cost of every track. A fast album click-through then tripped
        # TIDAL's rate limit, which suspends playback for a minute.
        fallback = _flac_fallback(session, track, requested, delivered)
        if fallback is not None:
            return fallback

    if (
        wants_lossless
        and not hi_res_blocked
        and delivered in _LOSSY_TIERS
        and requested != tidalapi.Quality.hi_res_lossless
    ):
        # The FLAC route was unavailable (entitlement, region, or no
        # FLAC asset). Hi-res is the v1 tier these sessions do serve,
        # so try it before accepting the AAC.
        session.config.quality = tidalapi.Quality.hi_res_lossless
        try:
            alt = track.get_stream()
        except Exception as exc:
            print(
                f"[tidal] hi-res upgrade failed for track "
                f"{getattr(track, 'id', '?')}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            alt = None
        finally:
            session.config.quality = requested
        alt_tier = delivered_quality(alt) if alt is not None else None
        if alt is not None and alt_tier in _LOSSLESS_TIERS:
            print(
                f"[tidal] track {getattr(track, 'id', '?')}: TIDAL served "
                f"{delivered!r} for a lossless request; upgraded to "
                f"{alt_tier!r}",
                file=sys.stderr,
                flush=True,
            )
            stream, delivered = alt, alt_tier

    return stream, stream.get_stream_manifest(), delivered


# Device-code (Limited Input Device) credentials. The selectable clients
# live in app/api_keys.py (tidekeeper's table); the user picks one on the
# login screen and it is persisted in settings.json. This pair is only
# the fallback for sessions minted before that choice existed: tokens are
# bound to their issuing client, so such a session must keep refreshing
# under the old TV client or TIDAL rejects it (4022).
_LEGACY_DEVICE_CODE_CLIENT_ID = "zU4XHVVkc2tDPo4t"
_LEGACY_DEVICE_CODE_CLIENT_SECRET = "VJKhDFqJPqvsPVNBV6ukXTJmwlvbttP7wlMlrc72se4="


def _apply_client(config: Any, key: api_keys.ApiKey) -> None:
    config.client_id = key.client_id
    config.client_secret = key.client_secret


def _selected_api_key() -> api_keys.ApiKey:
    """Client chosen on the login screen for the device-code flow.

    Read from disk rather than cached: the login UI can change it while
    the server is running, and a stale value would mint the token under
    the wrong client.

    A settings.json written by an older build can still name a retired
    client (the TV or legacy Fire TV pair). Those no longer complete a
    login, so they fall back to the hi-res default instead of minting a
    token nothing can play.
    """
    key = api_keys.get_item(load_settings().tidal_api_key_index)
    if key is None or key.index != api_keys.DEFAULT_API_KEY_INDEX:
        key = api_keys.get_item(api_keys.DEFAULT_API_KEY_INDEX)
    assert key is not None
    return key


def _selected_pkce_key() -> Optional[api_keys.ApiKey]:
    """Client chosen for the PKCE login.

    None means tidalapi's built-in mobile pair, which stays the default
    because it is the one known to be entitled for Max.
    """
    index = getattr(load_settings(), "tidal_pkce_client_index", 0)
    return api_keys.get_item(index)


class TidalClient:
    # Proactive refresh window: if the stored expiry is within this many
    # seconds of "now" when the watchdog ticks, refresh the token.
    # tidalapi tokens last an hour, so a 5-minute cushion gives us plenty
    # of margin without thrashing on refreshes.
    _REFRESH_WINDOW_SEC = 5 * 60
    # How often the watchdog re-checks. 60s is frequent enough that the
    # expiry window can't slip past between checks, but light enough on
    # cycles that nobody notices.
    _REFRESH_CHECK_INTERVAL_SEC = 60

    def __init__(self):
        config = tidalapi.Config(quality=tidalapi.Quality.high_lossless)
        # Device-code client picked on the login screen. Re-applied on
        # every load (see `_restore_device_client`) because a token can
        # only be refreshed by the client that minted it. The PKCE pair
        # is overridden separately by `_apply_pkce_client`.
        initial_key = _selected_api_key()
        self._device_code_index: Optional[int] = initial_key.index
        _apply_client(config, initial_key)
        # tidalapi's built-in PKCE pair, kept so a session minted under
        # an api_keys client can be reverted cleanly.
        self._pkce_default_client = (
            config.client_id_pkce,
            config.client_secret_pkce,
        )
        # Client index currently applied to the PKCE pair; None until
        # `_apply_pkce_client` runs (or the default is in force).
        self._pkce_client_index: Optional[int] = None
        # Client behind the cached authorization URL. The pasted code is
        # bound to it, so the exchange must use the same pair even if the
        # setting changed in between.
        self._pkce_url_client_index: Optional[int] = None
        # Client that minted the persisted PKCE token, written to the
        # session file so a relaunch refreshes under the same pair.
        self._session_pkce_client_index: Optional[int] = None
        self._pkce_url_cache: Optional[tuple[str, float]] = None
        self.session = tidalapi.Session(config)
        _install_tidal_request_gate(self.session)
        _swap_to_impersonated_transport(self.session)
        self._install_capturing_refresh()
        # Set by the server to its auth-cache invalidator. Invoked
        # when a hard refresh failure logs the user out, so the UI
        # bounces to Login immediately instead of after the cache TTL.
        self.on_auth_lost: Optional[Callable[[], None]] = None
        # Set by the server. Invoked when a deferred session load
        # finally succeeds, so the boot-time work that was skipped
        # because we looked signed out still happens.
        self.on_session_restored: Optional[Callable[[], None]] = None
        self._login_future: Optional[Future] = None
        # True when a persisted session exists on disk but the network
        # was unreachable when we tried to validate it, so
        # `session.user` / `session.session_id` were never populated.
        # Distinct from "no session" and from "Tidal rejected us": the
        # credentials are fine, we just haven't been able to use them
        # yet. Read by the server's auth check (offline ≠ signed out)
        # and cleared by the watchdog's retry once the network is back.
        self._session_load_deferred = False
        # One-shot latch so the watchdog's "this session can never be
        # renewed" warning is logged once rather than every tick.
        self._warned_no_refresh_token = False
        # Serializes the deferred-load retry across the watchdog tick
        # and connectivity-triggered retries, which can land together.
        self._deferred_retry_lock = threading.Lock()
        # Cached subscription-tier result. Populated on first call and
        # refreshed every _SUB_TTL_SEC. Subscription almost never changes
        # within a session, so a long TTL is fine.
        self._sub_cache: tuple[float, Optional[str]] = (0.0, None)
        self._sub_lock = threading.Lock()
        # Serializes EVERY token refresh — the watchdog's force_refresh
        # AND tidalapi's internal auto-refresh-on-401 — through one lock,
        # so concurrent 401s from parallel workers can't fire two
        # token_refresh RPCs at once. With rotating refresh tokens the
        # second POST would carry a token Tidal just invalidated and then
        # persist that broken lineage, logging the user out days later.
        # The single-flight check lives in _refresh_once.
        self._refresh_lock = threading.Lock()
        # Background watchdog: refreshes the token a few minutes before
        # it's due to expire. Without this, a long-running session (user
        # leaves the tab open overnight) lands a 401 the next time they
        # click anything, and though downstream callers have their own
        # retry-on-401 wrappers, it's a worse UX than just keeping the
        # token fresh in the background. Daemon so it doesn't block
        # process exit.
        self._refresh_stop = threading.Event()
        threading.Thread(
            target=self._refresh_watchdog, daemon=True, name="tidal-refresh"
        ).start()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _restore_device_client(self, stored_index: Any) -> None:
        """Re-pin the client that minted a persisted device-code token.

        TIDAL rejects a refresh under a different client id (4022), and
        the manifest gateway keys delivery (DRM vs. clear) on the same
        binding, so a session must keep the pair it was issued under.
        Files written before client selection existed carry no index;
        those tokens belong to the legacy TV client.
        """
        key = api_keys.get_item(stored_index)
        if key is not None:
            _apply_client(self.session.config, key)
            self._device_code_index = key.index
            return
        self.session.config.client_id = _LEGACY_DEVICE_CODE_CLIENT_ID
        self.session.config.client_secret = _LEGACY_DEVICE_CODE_CLIENT_SECRET
        # Remembered as "unknown" so the next save doesn't claim the
        # session belongs to the currently selected client.
        self._device_code_index = None

    def _apply_pkce_client(self, key: Optional[api_keys.ApiKey]) -> None:
        """Point tidalapi's PKCE pair at `key`; None restores mobile.

        Both the authorization request and the token exchange read
        `client_id_pkce`, so overriding it here makes the whole flow run
        under the chosen client. `client_enable_hires` then copies the
        pair onto the main credentials, which is what later API calls
        and the refresh path use.
        """
        config = self.session.config
        if key is None:
            config.client_id_pkce, config.client_secret_pkce = (
                self._pkce_default_client
            )
            return
        config.client_id_pkce = key.client_id
        config.client_secret_pkce = key.client_secret

    def _restore_pkce_client(self, stored_index: Any) -> None:
        """Re-pin the client that minted a persisted PKCE token.

        Same reasoning as `_restore_device_client`: the refresh token is
        only valid under its issuing client. Files written before the
        choice existed (and sessions left on the mobile default) carry
        no index and fall back to tidalapi's built-in pair.
        """
        key = api_keys.get_item(stored_index)
        self._apply_pkce_client(key)
        self._session_pkce_client_index = key.index if key is not None else None

    def load_session(self) -> bool:
        if not SESSION_FILE.exists():
            _tlog(f"load_session: no session file at {SESSION_FILE}")
            return False
        try:
            with open(SESSION_FILE) as f:
                data = json.load(f)
            expiry = (
                datetime.fromisoformat(data["expiry_time"])
                if data.get("expiry_time")
                else None
            )
            refresh_token = data.get("refresh_token")
            is_pkce = bool(data.get("is_pkce", False))
            refreshed_on_launch = False
            # For PKCE sessions, swap the client_id to the one that minted
            # the token BEFORE any Tidal call. Otherwise the /sessions call
            # inside load_oauth_session (and the proactive refresh below)
            # would run under the device-code client_id and Tidal would
            # reject the refresh (4022).
            if is_pkce:
                self.session.is_pkce = True
                # Restore the pair that minted the token before anything
                # hits Tidal, then mirror it onto the main credentials.
                self._restore_pkce_client(data.get("api_key_index"))
                self.session.client_enable_hires()
            else:
                self._restore_device_client(data.get("api_key_index"))
            # Proactive refresh: tidalapi only refreshes on 401 when the
            # response's userMessage starts with the exact string
            # "The token has expired." — Tidal has changed that message
            # format in the past, and when it doesn't match we get raw
            # 401s propagating to the caller (e.g. download endpoints).
            # If the stored expiry is in the past, skip trying the stale
            # token and refresh up front.
            if expiry and refresh_token and expiry <= _now_matching(expiry):
                # The stored access token is already expired, so this refresh
                # is what keeps a relaunch (app was closed past the token
                # lifetime) from bouncing the user to the login screen. It
                # must NOT swallow its failure: a transient error (network
                # not up yet at boot, 429, 5xx) has to defer-and-retry like
                # the offline path, or the fall-through to a stale-token
                # check_login() logs the user out for a blip. A genuine
                # rejection is a real re-login, but we log it so it stops
                # being invisible (#309).
                _transient = (TransientRefreshError,) + _NETWORK_ERRORS
                try:
                    if self._token_refresh_capturing(refresh_token):
                        self.save_session()
                        refreshed_on_launch = True
                        _tlog("load_session: expired token refreshed on launch")
                except tidalapi.exceptions.AuthenticationError as exc:
                    _tlog(
                        "load_session: stored refresh token rejected on "
                        f"launch — re-login required ({exc})"
                    )
                    # fall through: check_login() below returns False.
                except _transient as exc:
                    _tlog(
                        "load_session: launch refresh hit a transient error, "
                        f"keeping session to retry: {exc!r}"
                    )
                    self._session_load_deferred = True
                    return False
                except Exception as exc:
                    _tlog(f"load_session: unexpected launch-refresh error: {exc!r}")
                    # fall through rather than hide it.
            self.session.load_oauth_session(
                data["token_type"],
                self.session.access_token or data["access_token"],
                getattr(self.session, "refresh_token", None) or refresh_token,
                self.session.expiry_time or expiry,
                is_pkce=is_pkce,
            )
            # Reaching here means load_oauth_session got an answer from
            # Tidal, so whatever check_login() says is the real verdict
            # — including a False, which is a genuine rejection and not
            # something a retry would fix.
            self._session_load_deferred = False
            ok = self.session.check_login()
            if not ok:
                # The one logout that leaves no other trace: Tidal
                # rejects the stored session even though we either
                # refreshed it just now or judged it unexpired. Silence
                # here is what made #309 undiagnosable from a user's log.
                _tlog(
                    "load_session: Tidal rejected the stored session "
                    f"(stored expiry={expiry}, "
                    f"refreshed on launch={refreshed_on_launch})"
                )
            return ok
        except _NETWORK_ERRORS:
            # No route to Tidal. load_oauth_session never populated
            # session.user / session_id, which makes check_login()
            # return False from then on — with no exception for callers
            # to distinguish from a real rejection. Flag it so the auth
            # check reads this as "signed in, offline" and the watchdog
            # retries the load once the network is back. Without the
            # flag, launching offline 401s the local-library endpoints
            # and the session stays dead until the app restarts (#292).
            self._session_load_deferred = bool(refresh_token)
            return False
        except TidalBackoffError:
            # We refused to make the call ourselves — a rate-limit or
            # abuse-detection window is still open. Same situation as
            # no network: the credentials are untouched and the session
            # simply never got validated, so defer and retry rather
            # than reporting the user signed out for the rest of the
            # process. Backoff windows outlast a launch (30 minutes for
            # the abuse case), so this is reachable on a normal start.
            self._session_load_deferred = bool(refresh_token)
            return False
        except Exception as exc:
            # Anything left is a malformed or unreadable session file
            # (bad JSON, missing key, unparseable expiry). Treating that
            # as "not signed in" is right, but doing it silently means a
            # user who suddenly has to log in again finds nothing in the
            # log to explain why.
            _tlog(f"load_session: could not restore session: {exc!r}")
            return False

    def _token_refresh_capturing(self, refresh_token: str) -> bool:
        """Refresh the access token, keeping a rotated refresh token.

        tidalapi 0.8.11's `Session.token_refresh()` updates the
        access token, expiry, and token type from Tidal's response
        but never reads the `refresh_token` field of that response.
        Tidal rotates refresh tokens: a refresh sometimes comes back
        with a new refresh token and the old one is then invalidated
        server-side. Because tidalapi drops it, `save_session()` keeps
        re-persisting the original token; once Tidal kills it (a few
        days out) the next refresh fails and the user is logged out.

        This mirrors tidalapi's refresh exactly — same OAuth params,
        same config (so device-code and PKCE both stay correct), same
        impersonated transport — but also carries a rotated refresh
        token back onto the session so the subsequent save persists it.

        Failures are split by cause. A genuine auth rejection (Tidal's
        `invalid_grant` and friends, or a bare 401) raises
        AuthenticationError so the caller logs the user out. A transient
        transport failure (429 rate-limit, 5xx, non-JSON error page)
        raises TransientRefreshError so the caller keeps the session and
        retries later. Conflating the two used to delete a valid session
        whenever a background refresh landed on a rate-limit or outage.
        """
        session = self.session
        config = session.config
        is_pkce = bool(getattr(session, "is_pkce", False))
        params = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": (
                config.client_id_pkce if is_pkce else config.client_id
            ),
            "client_secret": (
                config.client_secret_pkce
                if is_pkce
                else config.client_secret
            ),
        }
        resp = session.request_session.post(config.api_oauth2_token, params)
        if resp.status_code != 200:
            # Split permanent auth rejection from transient transport
            # failure. Only the former should log the user out and wipe
            # the session file (force_refresh's AuthenticationError path).
            # A 429 rate-limit or a 5xx is "retry shortly" and the refresh
            # token is still valid — wrongly treating it as a dead token
            # is what silently logged users out days later when a
            # background refresh happened to hit a rate-limit window.
            try:
                body = resp.json()
            except ValueError:
                # Non-JSON body (e.g. an HTML 5xx error page). No OAuth
                # error code to read; treat as transient.
                body = {}
            error = str(body.get("error") or "")
            desc = body.get("error_description")
            # The OAuth spec's permanent failure codes, plus a bare 401
            # (bad client auth). Tidal returns `invalid_grant` when the
            # refresh token is expired or revoked — the genuine re-login
            # case. Anything else (429, 5xx, an unrecognized 4xx) is
            # transient: keep the session and let the watchdog retry.
            permanent = resp.status_code == 401 or error in (
                "invalid_grant",
                "invalid_request",
                "invalid_client",
                "unauthorized_client",
                "unsupported_grant_type",
            )
            if permanent:
                raise tidalapi.exceptions.AuthenticationError(
                    "Authentication failed with error "
                    f"'{error}: {desc}'"
                )
            raise TransientRefreshError(
                f"token refresh got HTTP {resp.status_code} "
                f"(error={error or 'none'}); refresh token kept"
            )
        body = resp.json()
        session.access_token = body["access_token"]
        # Naive UTC, matching how tidalapi stores expiry_time
        # everywhere else. Mixing a tz-aware value here would make
        # tidalapi's own naive-vs-aware datetime math raise.
        session.expiry_time = datetime.now(timezone.utc).replace(
            tzinfo=None
        ) + timedelta(seconds=body["expires_in"])
        session.token_type = body["token_type"]
        # The actual fix: a rotated refresh token must replace the
        # stored one. Tidal only returns this field when it rotates;
        # when absent the existing token stays valid, so keep it.
        new_refresh = body.get("refresh_token")
        if new_refresh:
            session.refresh_token = new_refresh
        return True

    def _refresh_once(self, based_on: Optional[str]) -> bool:
        """Locked, single-flight refresh shared by every runtime path
        (the watchdog's force_refresh and tidalapi's internal
        auto-refresh-on-401). Holding one lock across both is what
        actually delivers the "no two concurrent refresh RPCs"
        guarantee: previously only force_refresh took the lock, so two
        parallel 401s going through tidalapi's path both POSTed the
        same refresh token. With rotation, the second POST carries a
        token Tidal just invalidated, and persisting that broken
        lineage logs the user out a few days later.

        `based_on` is the refresh token the caller observed before
        contending for the lock. Once we hold the lock, if the
        session's refresh token has already moved on — another thread
        refreshed and Tidal rotated — that refresh supersedes this one,
        so we reuse it instead of re-POSTing the now-dead token.

        A permanent AuthenticationError propagates so force_refresh's
        logout-on-auth-failure path runs. A TransientRefreshError (429,
        5xx, network) is swallowed to a False return: the refresh token
        is still valid, so we keep the session and let the next watchdog
        tick retry rather than wiping a good session over a blip.
        """
        with self._refresh_lock:
            current = getattr(self.session, "refresh_token", None)
            if not current:
                return False
            if based_on is not None and current != based_on:
                # A parallel refresh already rotated the token we were
                # going to use; its result stands. Re-POSTing `current`
                # would be redundant, and re-POSTing `based_on` would
                # use a token Tidal already invalidated.
                return True
            try:
                ok = self._token_refresh_capturing(current)
            except TransientRefreshError as exc:
                _tlog(f"refresh: transient failure, session kept: {exc}")
                return False
            if ok:
                try:
                    self.save_session()
                except Exception:
                    # Best-effort: the in-memory session is already
                    # valid, and the watchdog / next refresh re-persists.
                    # Don't fail the refresh over a transient disk error.
                    pass
            return ok

    def _token_refresh_and_persist(self, refresh_token: str) -> bool:
        """token_refresh that captures rotation AND persists.

        This is what tidalapi's own request layer calls on a 401
        (request.py: `self.session.token_refresh(refresh_token)`).
        tidalapi's native implementation drops a rotated refresh
        token, so a few days later the next refresh fails and the
        user is silently logged out. Routing that internal path
        through _refresh_once captures the rotation, persists it, and
        — crucially — serializes it against every other refresh path
        so parallel 401s can't clobber each other's rotated token.
        """
        return self._refresh_once(based_on=refresh_token)

    def _install_capturing_refresh(self) -> None:
        """Shadow tidalapi's Session.token_refresh on this session
        with the rotation-capturing+persisting version, so EVERY
        refresh path — explicit force_refresh, the background
        watchdog, and tidalapi's own internal auto-refresh-on-401 —
        keeps a rotated refresh token. Re-applied after logout()
        because that builds a fresh Session. Same monkeypatch
        approach already used for the request gate and impersonated
        transport on this object."""
        self.session.token_refresh = self._token_refresh_and_persist

    def _notify_session_restored(self) -> None:
        """Tell the server a session that couldn't be validated at boot
        is now live, so the boot work that was skipped for a
        signed-out-looking session can run. Counterpart to
        _notify_auth_lost."""
        cb = getattr(self, "on_session_restored", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def _notify_auth_lost(self) -> None:
        """Tell the server its cached auth state is stale (refresh
        token dead, user logged out) so the next /auth/status flips
        to logged-out and the UI bounces to Login immediately."""
        cb = getattr(self, "on_auth_lost", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def force_refresh(self) -> bool:
        """Explicitly refresh the access token using the stored refresh
        token. Called by the download path when it hits a 401 — tidalapi's
        built-in refresh only triggers on a specific error message we
        can't rely on. Returns True on success.

        Logs to stderr on failure so the caller (and whoever is tailing
        the server log) can see whether the refresh actually ran and
        why it failed. If the refresh token is itself expired, wipes the
        persisted session file so the user is bounced to the login flow
        on next auth check instead of being stuck in a retry loop.

        Serialized on _refresh_lock so concurrent 401s from parallel
        workers don't fire two token_refresh RPCs at once. The second
        caller will see the fresh token after the first finishes.
        """
        refresh_token = getattr(self.session, "refresh_token", None)
        if not refresh_token:
            _tlog("force_refresh: no refresh_token on session")
            return False
        try:
            # _refresh_once takes _refresh_lock and single-flights: if a
            # parallel refresh already rotated the token, it reuses that
            # result instead of re-POSTing this now-stale one. It also
            # persists on success, so there's no save_session() here.
            ok = self._refresh_once(based_on=refresh_token)
        except Exception as exc:
            _tlog(f"force_refresh: token_refresh raised: {exc!r}")
            # AuthenticationError means the refresh token itself is
            # dead. Blow the session away so the user is prompted to
            # log in again rather than chasing 401s forever, and tell
            # the server to drop its cached auth state so the very
            # next /auth/status bounces the UI to Login immediately
            # instead of waiting out the cache TTL.
            if type(exc).__name__ == "AuthenticationError":
                try:
                    self.logout()
                except Exception:
                    pass
                self._notify_auth_lost()
            return False
        if not ok:
            _tlog("force_refresh: token_refresh returned False")
            return False
        _tlog("force_refresh: success")
        return True

    def session_load_deferred(self) -> bool:
        """True when a persisted session couldn't be validated because
        the network was down. The credentials are intact; we just
        haven't reached Tidal yet. Callers gating on "is this user
        signed in?" should treat this as yes-but-offline."""
        return self._session_load_deferred

    def retry_deferred_load_async(self) -> bool:
        """Retry a deferred session load on a background thread.

        Called when something observed connectivity returning, so
        recovery tracks the actual event instead of waiting out the
        watchdog's poll interval. Runs off-thread because the retry
        makes a Tidal round-trip and the caller is an HTTP handler.

        Returns True when a retry was started. No-ops when there's
        nothing deferred, or when a retry is already in flight — the
        browser fires `online` in bursts and each one must not stack
        another round-trip.
        """
        if not self._session_load_deferred:
            return False
        if self._deferred_retry_lock.locked():
            return False
        threading.Thread(
            target=self._retry_deferred_load,
            daemon=True,
            name="tidal-session-retry",
        ).start()
        return True

    def _retry_deferred_load(self) -> None:
        """Re-run the session load that a dead network aborted at boot.

        Until this succeeds, session.user and session.session_id are
        unset and every check_login() returns False, so the app would
        stay in its offline degraded state for the rest of the process
        even after the network came back.

        Serialized: the watchdog tick and a connectivity-triggered
        retry can land together, and two concurrent load_session calls
        would race on the same session object.
        """
        if not self._deferred_retry_lock.acquire(blocking=False):
            return  # another retry is already in flight
        try:
            if not self._session_load_deferred:
                return  # a racing retry already recovered it
            _tlog("session load was deferred (offline at boot) — retrying")
            if self.load_session():
                _tlog("deferred session load succeeded")
                self._notify_session_restored()
        finally:
            self._deferred_retry_lock.release()

    def _refresh_watchdog(self) -> None:
        """Background loop that refreshes the token before it expires.

        Sleeps for _REFRESH_CHECK_INTERVAL_SEC between checks. Each tick:
        retry a session load the network aborted at boot, then, if we're
        logged in and the stored expiry is within the window, fire
        force_refresh. Errors inside force_refresh are already logged
        there; the watchdog swallows its own errors so a transient
        network blip doesn't kill the thread.
        """
        while not self._refresh_stop.is_set():
            # Wake up early if someone signals stop (e.g. tests).
            if self._refresh_stop.wait(self._REFRESH_CHECK_INTERVAL_SEC):
                return
            try:
                if self._session_load_deferred:
                    self._retry_deferred_load()
                expiry = getattr(self.session, "expiry_time", None)
                refresh_token = getattr(self.session, "refresh_token", None)
                if not expiry or not refresh_token:
                    # A signed-in session with no refresh token can never
                    # be renewed, so this loop will skip it every tick
                    # until the access token expires and the user is
                    # silently signed out. Say so once, rather than once
                    # a minute, so the log explains the logout that is
                    # coming instead of only recording the aftermath.
                    if (
                        refresh_token is None
                        and expiry is not None
                        and not self._warned_no_refresh_token
                    ):
                        self._warned_no_refresh_token = True
                        _tlog(
                            "refresh watchdog: session has no refresh token, "
                            f"so it cannot be renewed and ends at {expiry} "
                            "(UTC) — a re-login will be required"
                        )
                    continue
                seconds_left = (expiry - _now_matching(expiry)).total_seconds()
                if seconds_left > self._REFRESH_WINDOW_SEC:
                    continue
                _tlog(
                    f"refresh watchdog: token expires in "
                    f"{seconds_left:.0f}s — refreshing"
                )
                self.force_refresh()
            except Exception as exc:
                _tlog(f"refresh watchdog: ignoring error: {exc!r}")

    def get_max_quality(self) -> Optional[str]:
        """Return the highest audio quality the logged-in account is
        allowed to stream, as a tidalapi Quality member name (e.g.
        'high_lossless'). None means the check failed or the user isn't
        logged in yet — caller should assume no filtering in that case.

        The subscription endpoint is the only source of truth for this;
        tidalapi itself doesn't surface the tier. Cached for _SUB_TTL_SEC
        so the frontend hitting /api/qualities on every mount doesn't
        fan out to a network call each time.
        """
        import sys as _sys

        now = time.monotonic()
        with self._sub_lock:
            cached_at, cached_val = self._sub_cache
            if now - cached_at < _SUB_TTL_SEC and cached_val is not None:
                return cached_val
        try:
            user = getattr(self.session, "user", None)
            uid = getattr(user, "id", None) if user else None
            if not uid:
                print(
                    "[tidal] get_max_quality: no user.id yet",
                    file=_sys.stderr,
                    flush=True,
                )
                return None
            resp = self.session.request.basic_request(
                "GET", f"users/{uid}/subscription"
            )
            if not resp.ok:
                print(
                    f"[tidal] get_max_quality: subscription fetch returned "
                    f"{resp.status_code}",
                    file=_sys.stderr,
                    flush=True,
                )
                return None
            data = resp.json()
        except Exception as exc:
            print(
                f"[tidal] get_max_quality: subscription fetch raised: {exc!r}",
                file=_sys.stderr,
                flush=True,
            )
            return None
        # Tidal exposes the ceiling under `highestSoundQuality`. Some
        # older/newer responses nest it under `subscription.type` —
        # check both and map to our internal code.
        hsq = (
            data.get("highestSoundQuality")
            or (data.get("subscription") or {}).get("highestSoundQuality")
            or (data.get("subscription") or {}).get("type")
            or ""
        )
        mapped = _SUB_QUALITY_MAP.get(str(hsq).upper())
        # No login-flow cap: entitlement follows the OAuth client, and
        # the Tidal HiRes device client (index 5) does serve
        # HI_RES_LOSSLESS via `playbackinfopostpaywall` — verified live
        # with a 24-bit/48 kHz DASH FLAC manifest. Clients that can't
        # serve the requested tier degrade themselves: the endpoint
        # answers 401/403 or downgrades to AAC, both handled in
        # `request_stream_manifest()`.
        print(
            f"[tidal] get_max_quality: raw={hsq!r} mapped={mapped!r} "
            f"response_keys={sorted(data.keys())}",
            file=_sys.stderr,
            flush=True,
        )
        if not mapped:
            return None
        with self._sub_lock:
            self._sub_cache = (now, mapped)
        return mapped

    def clamp_quality_to_subscription(
        self, requested: Optional[str]
    ) -> Optional[str]:
        """Downgrade `requested` to the highest tier the account can
        actually stream. Callers shove this in front of any code path
        that sets `session.config.quality` before a stream / download
        fetch — without it, picking e.g. "hi_res_lossless" on a HiFi
        account generates inevitable 401s from Tidal's playbackinfo
        endpoint. When the subscription lookup fails (network / stale
        token), pass the value through unchanged — better a 401 than
        silently downgrading the user for a transient lookup miss.
        """
        if not requested:
            return requested
        max_q = self.get_max_quality()
        if not max_q:
            return requested
        try:
            req_idx = _SUB_QUALITY_ORDER.index(requested)
            max_idx = _SUB_QUALITY_ORDER.index(max_q)
        except ValueError:
            return requested
        if req_idx <= max_idx:
            return requested
        return max_q

    def save_session(self):
        """Persist the session atomically and with 0600 perms.

        A crash between truncating the file and json.dump finishing would
        otherwise leave a zero-byte `tidal_session.json` and silently log
        the user out. We also chmod the file to user-only (the access +
        refresh tokens sit in plaintext) — otherwise on a shared Unix box
        every other user can read them.
        """
        expiry = self.session.expiry_time
        refresh_token = self.session.refresh_token
        if not refresh_token:
            # Never write away a refresh token we still hold on disk.
            #
            # It is the only credential that can renew a session, and an
            # absent value on the session object means "unchanged", not
            # "revoked": Tidal omits the field unless it rotates, and
            # tidalapi drops it when it does. `_token_refresh_capturing`
            # already applies that rule at the HTTP layer; persistence
            # has to apply it too or one save undoes the other.
            #
            # Writing the None instead disables renewal permanently. The
            # watchdog skips a session with no refresh token, so the
            # access token just expires and the user is signed out with
            # nothing in the log to explain it. That is how #309
            # reproduced here: a session file written with a null refresh
            # token and a four hour access token, signed out on the dot
            # when it expired.
            refresh_token = _stored_refresh_token()
            if refresh_token:
                _tlog(
                    "save_session: session had no refresh token, kept the "
                    "stored one"
                )
            else:
                _tlog(
                    "save_session: no refresh token to store — this session "
                    "cannot be renewed and will end when the access token "
                    "expires"
                )
        data = {
            "token_type": self.session.token_type,
            "access_token": self.session.access_token,
            "refresh_token": refresh_token,
            "expiry_time": expiry.isoformat() if expiry else None,
            # Track PKCE sessions separately — their client pair has to
            # be restored before load_oauth_session (and every refresh),
            # otherwise Tidal rejects the token under the wrong client.
            "is_pkce": bool(getattr(self.session, "is_pkce", False)),
            # Which client minted this token: the PKCE pair for browser
            # sessions (None = tidalapi's mobile default), the device-code
            # client otherwise. None for legacy files, which load_session
            # turns back into the legacy TV client.
            "api_key_index": (
                self._session_pkce_client_index
                if getattr(self.session, "is_pkce", False)
                else self._device_code_index
            ),
        }
        target = SESSION_FILE
        # Write to a sibling tempfile in the same dir so os.replace stays
        # atomic (rename across filesystems isn't).
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".tidal_session.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f)
            try:
                os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                # chmod is best-effort on Windows / exotic filesystems.
                pass
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def logout(self):
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
        # There is no longer a session to defer-load; leaving this set
        # would have the watchdog retry a file we just deleted and the
        # auth check report a signed-out user as offline-signed-in.
        self._session_load_deferred = False
        config = tidalapi.Config(quality=tidalapi.Quality.high_lossless)
        # The replacement session will sign in with whichever client the
        # login screen has selected.
        next_key = _selected_api_key()
        self._device_code_index = next_key.index
        _apply_client(config, next_key)
        self.session = tidalapi.Session(config)
        _install_tidal_request_gate(self.session)
        _swap_to_impersonated_transport(self.session)
        # logout() builds a fresh Session, so re-shadow token_refresh
        # or the next login's session would fall back to tidalapi's
        # rotation-dropping native implementation.
        self._install_capturing_refresh()
        # The subscription-tier cache belongs to the previous session.
        # Bust it so the next login re-detects under the new client_id.
        with self._sub_lock:
            self._sub_cache = (0.0, None)
        # The cached PKCE URL holds the OLD session's verifier. Without
        # clearing it, the next login returns a stale URL for up to 10 min
        # and its code exchange fails because the verifier is gone.
        self._pkce_url_cache = None
        self._pkce_url_client_index = None
        self._pkce_client_index = None
        # The replacement session has no PKCE token yet, so persistence
        # must stop claiming one.
        self._session_pkce_client_index = None

    # ------------------------------------------------------------------
    # OAuth login
    # ------------------------------------------------------------------

    def start_oauth_login(self) -> Tuple[str, str, Future]:
        """Initiate device-code OAuth. Returns (url, user_code, future).
        The future resolves when the user completes browser auth.

        Force the selected device-code client back into the config first.
        If a previous PKCE session was loaded at boot,
        `client_enable_hires` will have swapped `config.client_id` to
        the PKCE client — which Tidal rejects on the device-code
        endpoint with "Client is not a Limited Input Device client".
        """
        key = _selected_api_key()
        self._device_code_index = key.index
        _apply_client(self.session.config, key)
        login, future = self.session.login_oauth()
        self._login_future = future
        url = f"https://{login.verification_uri_complete}"
        return url, login.user_code, future

    def select_api_key(self, index: int) -> bool:
        """Choose the client used for the next device-code login.

        TIDAL binds a token to the client that minted it, and the
        manifest gateway also keys DRM vs. clear delivery on that
        binding, so a live device-code session can't adopt a new client:
        it is signed out and the caller must log in again. PKCE sessions
        keep their own hi-res client pair and are left running.

        Returns True when a re-login is required.
        """
        key = api_keys.get_item(index)
        if key is None:
            raise ValueError(f"Unknown TIDAL client index: {index}")
        current = load_settings()
        try:
            current_index = int(current.tidal_api_key_index)
        except (TypeError, ValueError):
            current_index = -1
        changed = current_index != key.index
        if changed:
            current.tidal_api_key_index = key.index
            save_settings(current)
        if bool(getattr(self.session, "is_pkce", False)):
            return False
        if not changed and self._device_code_index == key.index:
            return False
        self.logout()
        return True

    def select_pkce_api_key(self, index: int) -> bool:
        """Choose the client that mints the next PKCE token.

        `api_keys.PKCE_DEFAULT_INDEX` (0) keeps tidalapi's mobile pair.
        Returns True when an active PKCE session was signed out — its
        refresh token belongs to the client that issued it.
        """
        if not api_keys.is_pkce_index(index):
            raise ValueError(f"Unknown TIDAL client index: {index}")
        current = load_settings()
        try:
            current_index = int(current.tidal_pkce_client_index)
        except (TypeError, ValueError):
            current_index = -1
        if current_index != index:
            current.tidal_pkce_client_index = index
            save_settings(current)
        # The cached URL embeds the old client_id and its verifier is
        # consumed by the exchange, so make the next fetch generate a
        # fresh pair.
        self._pkce_url_cache = None
        if getattr(self.session, "is_pkce", False):
            self.logout()
            return True
        return False

    def complete_login(self, future: Future, timeout: int = 300) -> bool:
        """Block until the OAuth future resolves, then save session. Returns success."""
        try:
            future.result(timeout=timeout)
            if self.session.check_login():
                self.save_session()
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # PKCE login — required for hi-res (Max) downloads.
    #
    # Tidal caps the device-code OAuth `client_id` at Lossless regardless
    # of subscription. Only the PKCE flow (which uses a different
    # `client_id_pkce` baked into Tidal's own mobile app) is entitled to
    # stream hi-res FLAC. The tradeoff is UX: there's no device-code
    # handoff — the user logs in in a browser, gets redirected to an
    # "Oops" page (Tidal has no handler for our redirect), copies the
    # URL, and pastes it back to us.
    # ------------------------------------------------------------------

    def pkce_login_url(self) -> str:
        """URL the user should open in a browser to begin PKCE login.

        Cached for 10 minutes after generation. tidalapi's
        `session.pkce_login_url()` rotates the PKCE verifier on every
        call — if the frontend fetches the URL twice (React strict-
        mode double-render, a remount after coming back from Safari,
        a reconnect), the second call overwrites the first verifier,
        and the code the user pastes back (which was bound to the
        first verifier) fails to exchange with a 401. Caching the
        URL keeps the verifier stable across repeated fetches while
        one login attempt is in progress.
        """
        # The URL carries the client_id, so a selection change has to
        # invalidate the cached pair. Skip the lookup while a PKCE
        # session is live: its refresh uses this same pair, and swapping
        # it would get every background refresh rejected (4022).
        if not getattr(self.session, "is_pkce", False):
            key = _selected_pkce_key()
            index = (
                key.index if key is not None else api_keys.PKCE_DEFAULT_INDEX
            )
            if index != self._pkce_client_index:
                self._apply_pkce_client(key)
                self._pkce_client_index = index
                self._pkce_url_cache = None
            self._pkce_url_client_index = index
        cached = getattr(self, "_pkce_url_cache", None)
        if cached is not None:
            url, generated_at = cached
            if time.time() - generated_at < 600:  # 10 min
                return url
        url = self.session.pkce_login_url()
        self._pkce_url_cache = (url, time.time())
        return url

    def complete_pkce_login(self, redirect_url: str) -> tuple[bool, Optional[str]]:
        """Exchange the pasted 'Oops' redirect URL for access tokens,
        enable the hi-res client_id, and persist the session. Returns
        `(True, None)` on success, `(False, reason)` on failure so the
        API layer can surface a concrete error to the user.
        """
        # Respect any active backoff — pkce_get_auth_token hits
        # auth.tidal.com/v1/oauth2/token via its own requests.post so
        # the module-level request gate doesn't cover it. Refusing
        # here stops "Continue" spam from compounding a suspension.
        state = tidal_backoff_state()
        if state["active"]:
            return False, (
                f"Tidal is holding us off for another "
                f"{int(state['seconds_remaining'])}s ({state['reason']})"
            )
        try:
            # The pasted code is bound to the client that generated the
            # URL, so exchange under that pair. Falling back to the
            # stored selection covers a restart between the two steps
            # (the cached URL, and its verifier, are gone by then).
            index = self._pkce_url_client_index
            if index is None:
                key = _selected_pkce_key()
                index = (
                    key.index
                    if key is not None
                    else api_keys.PKCE_DEFAULT_INDEX
                )
            else:
                key = api_keys.get_item(index)
            self._apply_pkce_client(key)
            self._pkce_client_index = index
            self._session_pkce_client_index = (
                key.index if key is not None else None
            )
            token = self.session.pkce_get_auth_token(redirect_url)
            self.session.process_auth_token(token, is_pkce_token=True)
            # Defensive: make sure is_pkce actually sticks on the session
            # so save_session() writes is_pkce=True. If tidalapi's
            # process_auth_token ever stops setting this attribute, the
            # reloaded session would silently fall back to the non-hi-res
            # client_id on next restart.
            self.session.is_pkce = True
            # Swap the active client_id/secret to the PKCE pair the token
            # was minted under so subsequent API calls (and refreshes)
            # use the credentials that issued it.
            self.session.client_enable_hires()
            if self.session.check_login():
                self.save_session()
                # Bust the subscription cache — it may have been
                # populated earlier under a device-code session that
                # reported a Lossless ceiling; the PKCE session
                # should re-detect at Max.
                with self._sub_lock:
                    self._sub_cache = (0.0, None)
                # Clear the cached PKCE URL so a subsequent logout +
                # re-login generates a fresh verifier.
                self._pkce_url_cache = None
                return True, None
            return False, "check_login returned False"
        except Exception as exc:
            import sys as _sys
            # If tidalapi's requests call raised, pull the response
            # body out so we can see Tidal's actual error payload
            # ("invalid_grant", "invalid_client", code-already-used,
            # etc.) instead of just the 403 status.
            resp_body = ""
            resp = getattr(exc, "response", None)
            status_code = None
            if resp is not None:
                status_code = getattr(resp, "status_code", None)
                try:
                    resp_body = resp.text[:500]
                except Exception:
                    pass
            # tidalapi's pkce_get_auth_token uses its own requests.post
            # so the session-level gate never sees the response. Run
            # the shared classifier manually to keep the two paths in
            # lockstep.
            _classify_tidal_error(status_code, resp_body)
            print(
                f"[tidal] complete_pkce_login failed: {exc!r}\n"
                f"  redirect_url: {redirect_url[:200]}\n"
                f"  response_body: {resp_body}",
                file=_sys.stderr,
                flush=True,
            )
            friendly = _friendly_pkce_error(exc)
            if friendly:
                return False, friendly
            detail = f"{type(exc).__name__}: {exc}"
            if resp_body:
                detail = f"{detail} | body: {resp_body}"
            return False, detail

    # ------------------------------------------------------------------
    # User info
    # ------------------------------------------------------------------

    def get_user_info(self) -> Optional[str]:
        try:
            user = self.session.user
            for attr in ("username", "first_name", "email"):
                value = getattr(user, attr, None)
                if value:
                    return str(value)
            return "Tidal User"
        except Exception:
            return None

    def get_user_avatar_url(self) -> Optional[str]:
        """Tidal exposes a per-user `picture_id` on FetchedUser; most
        accounts don't set one. Returns a CDN URL (210×210) if available,
        or None — callers should fall back to initials.
        """
        try:
            user = self.session.user
            img = getattr(user, "image", None)
            if callable(img):
                return img(210)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # URL parsing and content fetching
    # ------------------------------------------------------------------

    def parse_url(self, url: str) -> Tuple[str, str]:
        """Return (content_type, id_string) for a Tidal URL."""
        patterns = {
            "track": r"/track/(\d+)",
            "album": r"/album/(\d+)",
            "playlist": r"/playlist/([\w\-]+)",
        }
        for content_type, pattern in patterns.items():
            m = re.search(pattern, url)
            if m:
                return content_type, m.group(1)
        raise ValueError(f"Unrecognized Tidal URL: {url}")

    def fetch_url(self, url: str):
        """Fetch track, album, or playlist object from a Tidal URL.
        Returns (content_type, object)."""
        content_type, content_id = self.parse_url(url)
        if content_type == "track":
            return content_type, self.session.track(int(content_id))
        if content_type == "album":
            return content_type, self.session.album(int(content_id))
        if content_type == "playlist":
            return content_type, self.session.playlist(content_id)
        raise ValueError(f"Unsupported content type: {content_type}")

    def get_track_url(self, track) -> Optional[str]:
        try:
            return track.get_url()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 25) -> dict:
        """Returns dict with keys: tracks, albums, artists, playlists."""
        models = [tidalapi.Track, tidalapi.Album, tidalapi.Artist, tidalapi.Playlist]
        return self.session.search(query, models=models, limit=limit)

    # ------------------------------------------------------------------
    # Library / favorites
    # ------------------------------------------------------------------

    def _favorites_sorted(self, method_name: str) -> list:
        """Fetch every favorite of the given type, paging through the API if
        necessary, then sort newest-first."""
        try:
            method = getattr(self.session.user.favorites, method_name)
        except Exception:
            return []

        items = _fetch_all_pages(method)
        return _sort_by_date_added(items)

    def get_favorite_artists(self) -> list:
        return self._favorites_sorted("artists")

    def get_favorite_tracks(self) -> list:
        return self._favorites_sorted("tracks")

    def get_favorite_albums(self) -> list:
        return self._favorites_sorted("albums")

    def get_favorite_playlists(self) -> list:
        return self._favorites_sorted("playlists")

    def get_user_playlists(self) -> list:
        # A deferred session (no network, or a Tidal backoff window at
        # boot) never populates `session.user`, so this used to raise
        # AttributeError straight through /api/playlists/mine as a 500.
        # Degrade the same way `_favorites_sorted` does: no user, no
        # favourites to list.
        user = getattr(self.session, "user", None)
        if user is None:
            return []
        items = _fetch_all_pages(user.playlists)
        return _sort_by_date_added(items)

    def get_artist_albums(self, artist) -> list:
        # No `limit`, which is tidalapi's "everything": Tidal returns
        # the artist's whole album list in one response (168 for Taylor
        # Swift, measured) rather than a first page. Don't add a cap
        # here — a number chosen to look generous is what truncated the
        # EPs shelf for prolific artists.
        try:
            return list(artist.get_albums())
        except Exception:
            return []

    def get_artist_releases(self, artist, limit: int = 30) -> list:
        """Albums + EPs + singles for an artist, combined. `get_albums()`
        alone excludes singles, which is exactly the content the feed
        page needs (singles drop more frequently than albums). Results
        are deduped by id, sorted nothing — the caller sorts by date."""
        out: list = []
        seen: set = set()
        for fn in (
            lambda: artist.get_albums(limit=limit),
            lambda: artist.get_ep_singles(limit=limit),
        ):
            try:
                for item in fn():
                    key = str(getattr(item, "id", "") or "")
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
            except Exception:
                continue
        return out

    def get_artist_top_tracks(self, artist) -> list:
        """Top tracks for an artist, with one retry and an album-tracks
        fallback. Tidal's get_top_tracks is intermittently flaky for
        very popular artists (Travis Scott, Drake, Taylor Swift in
        casual testing) — the request succeeds but returns an empty
        list ~30% of the time. Empty top tracks cascade into broken
        Spotify monthly-listeners on the artist page (no ISRC to
        pivot on), so we retry once, then fall back to walking the
        first couple of albums for tracks if the API still gives us
        nothing.
        """
        for _ in range(2):
            try:
                tracks = list(artist.get_top_tracks(limit=10))
                if tracks:
                    return tracks
            except Exception:
                pass

        # Fallback: pull tracks off the first couple of releases.
        # Anything with an ISRC works for the artist-resolve pivot
        # downstream, and a couple of album cuts is plenty.
        try:
            albums = list(artist.get_albums(limit=2))
        except Exception:
            albums = []
        out: list = []
        for alb in albums:
            try:
                out.extend(list(alb.tracks())[:5])
            except Exception:
                continue
            if len(out) >= 10:
                break
        return out

    def get_album_tracks(self, album) -> list:
        try:
            return list(album.tracks())
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Writes (favorites + playlists)
    # ------------------------------------------------------------------

    @property
    def user_id(self) -> Optional[str]:
        try:
            uid = getattr(self.session.user, "id", None)
            return str(uid) if uid is not None else None
        except Exception:
            return None

    def favorite(self, kind: str, obj_id: str, add: bool) -> None:
        """Add or remove a favorite. `kind` in {track, album, artist, playlist, mix}."""
        # AttributeError on `None.favorites` reads like a bug in our code;
        # this is really "the session hasn't finished loading yet".
        user = getattr(self.session, "user", None)
        if user is None:
            raise RuntimeError("Tidal session isn't ready yet — sign-in is still pending")
        favs = user.favorites
        # tidalapi's mix helpers are plural (add_mixes / remove_mixes) and
        # accept a list of string ids, while the other kinds are singular
        # and expect an int (or a uuid string for playlist).
        if kind == "mix":
            method_name = ("add_" if add else "remove_") + "mixes"
            method = getattr(favs, method_name, None)
            if method is None:
                raise ValueError("Unsupported favorite kind: mix")
            method([str(obj_id)])
            return
        method_name = ("add_" if add else "remove_") + kind
        method = getattr(favs, method_name, None)
        if method is None:
            raise ValueError(f"Unsupported favorite kind: {kind}")
        # tidalapi: playlist uses uuid (str), others use int.
        arg: object = obj_id if kind == "playlist" else int(obj_id)
        method(arg)

    def favorites_snapshot(self) -> dict:
        """Return sets of favorite IDs the UI needs to render heart states."""
        result = {"tracks": [], "albums": [], "artists": [], "playlists": [], "mixes": []}
        for kind, attr in (
            ("tracks", "get_favorite_tracks"),
            ("albums", "get_favorite_albums"),
            ("artists", "get_favorite_artists"),
            ("playlists", "get_favorite_playlists"),
        ):
            try:
                items = getattr(self, attr)()
                result[kind] = [
                    str(getattr(i, "id", "") or getattr(i, "uuid", "")) for i in items
                ]
            except Exception:
                continue
        # Mixes live under session.user.mixes() in tidalapi rather than
        # self.get_favorite_* like the other kinds.
        try:
            mixes = self.session.user.mixes(limit=200)
            result["mixes"] = [str(getattr(m, "id", "") or "") for m in mixes if getattr(m, "id", "")]
        except Exception:
            pass
        return result

    def create_playlist(self, title: str, description: str = ""):
        user = getattr(self.session, "user", None)
        if user is None:
            raise RuntimeError("Tidal session isn't ready yet — sign-in is still pending")
        return user.create_playlist(title, description)

    def owns_playlist(self, playlist) -> bool:
        try:
            creator = getattr(playlist, "creator", None)
            if creator is None:
                return False
            return str(getattr(creator, "id", "")) == (self.user_id or "")
        except Exception:
            return False

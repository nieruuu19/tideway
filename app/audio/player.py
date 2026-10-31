"""PCMPlayer — PyAV + sounddevice audio engine.

Decodes DASH / local audio with PyAV and drives a sounddevice
OutputStream at the track's native sample rate + native format.
No software resampling or format conversion anywhere in the
playback path; samples hit CoreAudio / WASAPI / ALSA bit-identical
to what's in the source.

Gapless transitions: preloaded next-track PCM is spliced into the
live OutputStream at the sample boundary (same-rate) or triggers a
~50ms stream reopen (cross-rate). Both paths sit behind `preload()`
and fire automatically on natural end-of-track.

Threading model:
  - Main thread (HTTP handlers): calls load/play/pause/resume/seek/
    stop/set_volume/set_muted. Mutates state through a lock; never
    blocks on the audio pipeline.
  - Decoder thread: pulls PCM chunks from `Decoder`, pushes into
    `_pcm_queue`. Exits at EOF or when `_stop_flag` is set. Seek
    is applied in-flight: `seek()` calls `Decoder.request_seek()`
    + drains the queue, and the decoder picks up the seek on its
    next iteration.
  - Audio callback (sounddevice realtime thread): drains the queue,
    fills output buffers. Honours the `_paused` flag (outputs
    silence, doesn't drain) and applies volume/mute.
"""
from __future__ import annotations

import faulthandler
import json
import logging
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import sounddevice as sd  # type: ignore
import tidalapi

from app import atmos as atmos_mod
from app.audio import pa_guard
from app.http import STREAM_SESSION
from app.audio.decoder import Decoder
from app.audio.crossfade import mix_crossfade_block
from app.audio.crossfeed import Crossfeed
from app.audio.eq import (
    Equalizer,
    ParametricBand,
    build_parametric_sos,
    parametric_band_from_dict,
    parametric_preset,
)
from app.audio.manifest_cache import ManifestCache
from app.audio.replaygain import (
    ReplayGain,
    ReplayGainMode,
    ReplayGainTags,
    TIDAL_TAG_REFERENCE_LUFS,
    VALID_MODES as REPLAYGAIN_VALID_MODES,
    compute_gain_db as compute_replaygain_db,
    compute_measured_gain_db,
)
from app.audio.output_devices import (
    list_output_devices,
    resolve_fallback_output_device,
    resolve_output_device,
    resolve_surround_output_device,
)
from app.audio import native_atmos
from app.audio import loudness
from app.audio.atmos_dash import DashPublication, dash_server
from app.audio.native_atmos import NativeAtmosPlayer
from app.audio.segment_reader import SegmentReader
from app.audio.stream_cache import StreamCache
from app.audio.win_com import ensure_com_initialized
from app.paths import user_data_dir

# Optional audio output taps. Imported at module load (not lazily
# from inside the audio callback) because Python's `import` statement
# acquires the import lock to look up an already-cached module, and
# during a concurrent cold-load elsewhere in the process — Spotify
# enrichment first-loading spotapi when the user opens an artist page
# is the canonical case — that lock can be held for hundreds of ms.
# A lazy `import` inside the callback would block on it for the same
# duration and starve the realtime audio thread. The user hears that
# as stutter and clipping during cold artist-page navigation. With
# the imports here, the callback does plain attribute reads on
# already-loaded modules; the import lock is never touched on the
# realtime path. If the optional dep is missing on this machine
# (pyatv not installed, etc.) the reference stays None and the
# corresponding tap branch is skipped.
try:
    from app.audio import cast as _cast_mod
except Exception:  # noqa: BLE001
    _cast_mod = None  # type: ignore[assignment]
try:
    from app.audio.upnp import upnp_manager as _upnp_manager
except Exception:  # noqa: BLE001
    _upnp_manager = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Persistent audio-event log. A Finder-launched .app discards
# stdout/stderr, so the [audio]/[pcm] prints below never survive a
# real incident — an audio glitch was undebuggable from logs. This
# rotating file in the app data dir does survive, mirroring how
# window_chrome keeps its own. Realtime-safe: the callback emitters
# are rate-limited to ~1/sec and the swap line fires once per track
# change, so no per-block file I/O is added. Falls back to a
# NullHandler if the data dir isn't writable, so logging never raises
# on the audio thread.
audio_log = logging.getLogger("tideway.audio")
audio_log.setLevel(logging.INFO)
audio_log.propagate = False
if not audio_log.handlers:
    try:
        from logging.handlers import RotatingFileHandler
        from app.paths import user_data_dir

        _ah = RotatingFileHandler(
            str(user_data_dir() / "audio.log"),
            maxBytes=1_000_000,
            backupCount=3,
        )
        _ah.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        audio_log.addHandler(_ah)
    except Exception:
        audio_log.addHandler(logging.NullHandler())


# Back-pressure between the decoder and the audio callback. Each
# item is one AudioFrame's worth of PCM (~1024 samples typical).
# Sized to hold ~2s of audio at 44.1k, ~1s at 96k, ~0.5s at 192k —
# enough headroom that a single slow segment fetch (TLS handshake,
# CDN edge swap, brief network blip) drains the queue without
# starving the audio callback. Earlier sizing of 30 was tight for
# CD-rate (0.7s) and dangerously tight for hi-res (0.16s at 192k),
# which produced audible stutter on the slower segment fetches
# hi-res implies (3-7 MB per segment vs. 0.5-1 MB for lossless).
# On older driver stacks (Windows 10 LTSB) sustained underruns
# can escalate to a fatal native abort; the bigger queue is the
# real fix and the diagnostic logging in _audio_callback is the
# safety net.
# Memory cost is negligible (~800 KB at int32 stereo).
_PCM_QUEUE_MAX = 100

# Minimum number of decoded chunks the incoming (preloaded) track must
# have buffered before a crossfade may begin, so the fade-in side doesn't
# immediately starve. If the preload isn't this far ahead yet, the
# crossfade simply doesn't start this callback and we re-check on the
# next one (and if the window closes first, it falls back to the gapless
# cut). A handful of chunks is a fraction of a second of audio.
_XFADE_MIN_PREBUFFER_CHUNKS = 4

# Hard ceiling on the two Tidal round-trips in _resolve_source. The
# transport already bounds each request (app.http.DEFAULT_TIMEOUT,
# ~36s worst case), so this is a backstop: it guarantees the player
# pipeline lock is released even if a transport ever fails to honour
# its own timeout. Sits above the transport ceiling so the
# transport's own, more specific error surfaces first in the normal
# case. Without it, one stalled request held the pipeline lock
# forever and every playback control hung until an app restart.
_RESOLVE_TIMEOUT_S = 45.0

# How long a caller waits for a resolve another thread already started
# for the same track (a hover prefetch the click caught mid-flight)
# before giving up and doing the work itself. Several times a healthy
# resolve, short enough that a wedged one cannot hold a click hostage:
# past this the caller resolves on its own, which only costs the
# duplicate work this was meant to save.
_RESOLVE_JOIN_S = 4.0

# How long the play path waits for a hover prefetch's init + first
# segment fetches before pulling them itself. Those fetches take about
# a second from cold; a click that joins one late waits the remainder,
# which still beats a second, competing download.
_WARM_BYTES_JOIN_S = 3.0

# How much of a single-URL (progressive AAC) source a warm fetches.
# Enough for the MP4 header plus ~25 s of 320 kbps audio; SegmentReader
# streams the remainder when the track is actually played. Downloads
# the whole file for every hovered track would pull tens of MB on an
# album-mount warm and compete with the audio callback.
_WARM_HEAD_BYTES = 1024 * 1024

# While DLNA is casting a bounded per-track file, the local decoder waits
# this long for that file's whole-track demux to land on disk before it
# opens it. Bounded by the same transport budget as a resolve; the file
# only has to cross the LAN, and the alternative (a second TIDAL
# connection for the same track) trips TIDAL's one-track-at-a-time limit.
_CAST_FILE_READY_S = 60.0

# How long a load may sit in "loading" before the watchdog calls it
# wedged. Kept below _CAST_FILE_READY_S, but the cast demux wait is
# excluded from this timer explicitly (see _tick_load_stall) since it
# is bounded on its own and legitimately exceeds this window for long,
# hi-res tracks.
_LOAD_STALL_S = 30.0

# Every stream hand-off in this module goes through pa_guard: closing an
# OutputStream is not atomic and terminating PortAudio invalidates live
# streams, both of which crashed the backend inside libportaudio when
# the finished callback and device recovery raced each other (see
# pa_guard's docstring).
_close_pa_stream = pa_guard.close_stream


@dataclass
class StreamInfo:
    """What's actually audible — drives the now-playing quality badge."""
    source: str  # "stream" | "local"
    codec: Optional[str] = None
    bit_depth: Optional[int] = None
    sample_rate_hz: Optional[int] = None
    audio_quality: Optional[str] = None
    audio_mode: Optional[str] = None
    # ReplayGain metadata pulled from the tidalapi Stream object.
    # Tidal masters come pre-tagged with both track-level and album-
    # level loudness offsets (relative to the EBU R128 reference) and
    # the actual sample peaks. The audio engine uses these to apply
    # loudness leveling at playback time when the user enables it.
    # All four are Optional because not every stream carries them
    # (older catalog, low-quality tier, edge cases).
    track_replay_gain_db: Optional[float] = None
    track_peak: Optional[float] = None
    album_replay_gain_db: Optional[float] = None
    album_peak: Optional[float] = None


_LOSSLESS_REQUEST_TIERS = frozenset({"high_lossless", "hi_res_lossless"})


def _cached_tier_is_usable(
    quality: Optional[str], info: Optional[StreamInfo]
) -> bool:
    """Whether a cache entry may satisfy this quality request.

    TIDAL used to answer AAC for every lossless tier (see
    `app.tidal_client._flac_fallback`), and those wrong-tier tracks are
    still in the manifest cache and in the disk cache under their
    lossless keys. Replaying one would pin the badge — and the audio —
    to Low forever, so a lossy entry is treated as a miss and the
    track is re-resolved through the FLAC path.
    """
    if not quality or quality not in _LOSSLESS_REQUEST_TIERS or info is None:
        return True
    raw = getattr(info, "audio_quality", None)
    if raw is None:
        return True
    return str(getattr(raw, "value", raw)).upper() not in ("LOW", "HIGH")


@dataclass
class _Preload:
    """A pre-decoded next track. Created by `preload()` while the
    current track still has time to play. At end-of-current-track
    the audio callback tries to splice this in without pausing the
    OutputStream — if `sample_rate` and `sd_dtype` match the
    current stream, the swap is sample-accurate and the user hears
    no gap between tracks.
    """
    track_id: str
    quality: Optional[str]
    duration_ms: int
    stream_info: StreamInfo
    source_urls: Optional[list[str]]
    source_path: Optional[str]
    decoder: Optional[Decoder] = None
    queue: "Optional[queue.Queue[Optional[np.ndarray]]]" = None
    thread: Optional[threading.Thread] = None
    stop_flag: Optional[threading.Event] = None
    done: Optional[threading.Event] = None
    sample_rate: int = 0
    channels: int = 0
    sd_dtype: str = ""
    # Metadata for THIS preloaded track. Carried on the preload so the
    # gapless swap notifies the renderer with the track actually being
    # adopted, not the shared _current_track_meta slot (which the racing
    # desktop clock may have clobbered with a later track's metadata).
    track_meta: Optional[dict] = None
    # Cast/bounded-DLNA preload: a metadata-only record (no local decoder
    # or PCM queue). The renderer streams the bounded temp file; this
    # record only lets the renderer-clock watchdog adopt the track the
    # renderer already auto-advanced into (see on_renderer_track_ended).
    cast: bool = False


@dataclass
class PlayerSnapshot:
    state: str  # idle | loading | playing | paused | ended | error
    track_id: Optional[str]
    position_ms: int
    duration_ms: int
    volume: int  # 0..100
    muted: bool
    error: Optional[str] = None
    seq: int = 0
    stream_info: Optional[StreamInfo] = None
    # How Dolby content is actually reaching the listener: "atmos" when
    # the JOC objects were decoded and placed by a Dolby Atmos spatial
    # renderer, "dolby_audio" when only the 5.1/7.1 bed is audible,
    # None for non-Dolby content. Computed per snapshot (not stored on
    # StreamInfo) because a Spatial Sound change mid-track changes it.
    immersive: Optional[str] = None
    # Force Volume: when true the volume slider in the UI should
    # render disabled (value pinned at 100) because the backend
    # rejects changes while it's on.
    force_volume: bool = False
    # Why playback stopped, when that matters to the UI. "backoff" means
    # Tidal's request gate is closed, so the failure says nothing about
    # this track: the queue must not skip ahead, or it walks every
    # remaining track into the same wall and lengthens the strike.
    # None for ordinary errors, which are track-specific and skippable.
    error_kind: Optional[str] = None


def _tidal_backoff_active() -> bool:
    """Whether the Tidal request gate is closed right now.

    Imported inside the function because player.py deliberately doesn't
    pull in the Tidal client at module scope (its request gate installs
    itself on the session at import time)."""
    try:
        from app.tidal_client import tidal_backoff_state

        return bool(tidal_backoff_state()["active"])
    except Exception:
        return False


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Best-effort detection of a Tidal auth (expired-token) error.

    Mirrors server._looks_like_401 deliberately — duplicated rather
    than imported because player.py must not import server (server
    imports player). tidalapi wraps these as requests.HTTPError with
    .response.status_code 401/403, or surfaces them as a RuntimeError
    whose str() carries '401' / 'Unauthorized'."""
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) in (401, 403):
        return True
    msg = str(exc)
    return "401" in msg or "Unauthorized" in msg


def _resolve_surround_channels(
    source_channels: int,
    device_max_channels: Optional[int],
    *,
    windows: bool,
    exclusive: bool,
) -> int:
    """How many channels to feed the output stream for a surround source.

    Windows Spatial Sound (Sonic / Dolby Atmos for Headphones) is a
    shared-mode APO that virtualises the endpoint's bed. It needs the
    full bed to place objects: a 5.1 Atmos core padded up to the
    endpoint's advertised 7.1 gives the virtualiser eight channels to
    work with, and the OS does the headphone downmix plus binaural
    rendering. Feeding it 5.1 (or stereo) just plays the three front
    channels and skips the spatial stage.

    Exclusive mode bypasses the APO chain (bit-perfect output), so it
    never pads and only downmixes when the device genuinely can't take
    the source layout. A device that reports stereo has spatial off or
    isn't a spatial endpoint; padding can't help there, so that case
    falls through to the existing downmix.

    Returns the channel count to request from the decoder/stream.
    """
    if device_max_channels is None:
        return source_channels
    if device_max_channels < source_channels:
        # Shared backends report the OS mixer config; when it is below
        # the source we must downmix or the open fails.
        return device_max_channels if device_max_channels >= 6 else 2
    if (
        windows
        and not exclusive
        and source_channels >= 6
        and device_max_channels in (6, 8)
        and device_max_channels > source_channels
    ):
        return device_max_channels
    return source_channels


class PCMPlayer:
    def __init__(
        self,
        session_getter: Callable[[], tidalapi.Session],
        local_lookup: Optional[Callable[[str], Optional[str]]] = None,
        quality_clamp: Optional[Callable[[str], Optional[str]]] = None,
        force_refresh: Optional[Callable[[], bool]] = None,
        native_atmos_enabled: Optional[Callable[[], bool]] = None,
        native_atmos_dash: Optional[Callable[[], bool]] = None,
    ):
        self._session_getter = session_getter
        self._local_lookup = local_lookup
        self._quality_clamp = quality_clamp
        # Called when a playback resolve hits an expired Tidal token.
        # Returns True if the token was refreshed (caller retries
        # once), False if the refresh token itself is dead (caller
        # surfaces a clear "log in again"). Injected by the server
        # the same way session_getter is; None in tests/standalone.
        self._force_refresh = force_refresh
        # Local Atmos files can be handed to Windows' own pipeline
        # (Dolby decoder MFT + spatial renderer) instead of PyAV,
        # which only renders the core bed. The predicate is injected
        # so the player doesn't import settings; None = always off,
        # which is what standalone/test constructions want.
        self._native_atmos_enabled = native_atmos_enabled
        # When on, a streamed Atmos track is handed to MediaPlayer as a
        # DASH presentation served over loopback while it downloads, so
        # playback starts on the first segment instead of waiting for
        # the whole 20-30 MB asset (see atmos_dash.py). A stream the
        # helper refuses falls back to the finished file.
        self._native_atmos_dash = native_atmos_dash

        self._lock = threading.RLock()
        # Serializes load/stop/seek/preload so concurrent HTTP calls
        # can't race each other into opening overlapping streams or
        # sharing a single decoder across threads. Held for the full
        # duration of a state-changing operation; the audio callback
        # and decoder threads never touch it. RLock so play_track()
        # can take the lock around its load() + play() pair without
        # needing a separate locked helper for play.
        self._pipeline_lock = threading.RLock()
        self._state = "idle"
        self._current_track_id: Optional[str] = None
        self._current_duration_ms: int = 0
        self._current_stream_info: Optional[StreamInfo] = None
        self._last_error: Optional[str] = None
        self._seq = 0
        self._listeners: list[Callable[[PlayerSnapshot], None]] = []

        self._decoder: Optional[Decoder] = None
        self._stream: Optional[sd.OutputStream] = None
        self._stream_sample_rate: Optional[int] = None
        self._pcm_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=_PCM_QUEUE_MAX
        )
        self._decoder_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._decoder_done = threading.Event()
        self._callback_carry: Optional[np.ndarray] = None
        self._samples_emitted = 0

        self._paused = False
        # Bumped by pause() every time the user actually asks for a
        # pause. play_track() samples it either side of the load so a
        # pause pressed while a track is still resolving isn't lost:
        # both _load_locked and play() clear _paused as part of normal
        # startup, so without a record of the request there is nothing
        # left to tell "user paused mid-load" from "fresh load".
        self._pause_requests = 0
        self._seeking = False
        # Set while _swap_pipeline_to() is rewriting the pipeline refs
        # from another thread. The audio callback is lock-free, so
        # without this it can run mid-swap and apply the previous
        # track's ReplayGain/filter state to the new track's samples —
        # a split second of full-scale clipped audio on track change.
        # Callback emits silence while this is set, same as _seeking.
        self._swapping = False
        self._volume = 100  # 0..100
        self._muted = False
        # External output active: when something else is rendering
        # the audio (Cast device, Tidal Connect target), the local
        # sounddevice OutputStream still runs but
        # writes silence so the user doesn't hear two copies of
        # the music coming from their Mac speakers and the remote
        # device. The PCM tap to those external sinks happens BEFORE
        # this silencing (see audio_callback) so the receivers still
        # get full-amplitude audio at their own volume control.
        # Flag is owned by the various external-output managers
        # (cast.py + tidal_connect.py both flip it on connect /
        # disconnect via the registered silencer hook).
        self._external_output_active: bool = False
        # sounddevice device-index string ("" = system default).
        # Applied on the next _open_output_stream(). Device-switch
        # live is a Phase 6 refinement.
        self._selected_device_id: str = ""
        # Exclusive Mode: request bit-perfect output from the OS audio
        # layer. Applied on the next _open_output_stream() so toggling
        # mid-track waits for the next stream open.
        self._exclusive_mode: bool = False
        # Force Volume: pin software volume at 100; set_volume becomes
        # a no-op while this is on. User attenuates via DAC/OS instead
        # so bit-depth isn't scaled away before the stream leaves us.
        self._force_volume: bool = False
        # Set True around any code path that intentionally stops the
        # current OutputStream because it's about to open a fresh
        # one in its place (set_output_device, exclusive-mode flip,
        # cross-rate gapless bridge). The finished_callback that
        # sounddevice fires from `stream.stop()` checks this flag and
        # skips its end-of-track / device-loss logic so a transient
        # stream replacement doesn't show up to the frontend as
        # `state="ended"` or trigger device-loss recovery against
        # a perfectly healthy device.
        self._replacing_stream: bool = False
        # Rate limit for the watchdog's dead-stream recovery. A Windows
        # Spatial Sound change can kill the stream without PortAudio
        # ever calling finished_callback, which would otherwise leave
        # the player reporting "playing" with silence forever.
        self._last_watchdog_recover_t: float = 0.0

        # Remembered source for restart-based seek. For DASH streams
        # we store the full URL list (index 0 = init segment, 1+ =
        # media segments) so seek can build a new SegmentReader
        # starting at the nearest segment. For local files we store
        # the filesystem path. `None` when no track is loaded.
        self._source_urls: Optional[list[str]] = None
        self._source_path: Optional[str] = None
        self._current_track_meta: Optional[dict] = None

        # Sounddevice dtype the current OutputStream opened with.
        # Stored so the gapless-swap check can verify the preloaded
        # track matches.
        self._stream_sd_dtype: Optional[str] = None
        self._stream_channels: Optional[int] = None
        # What the last successful output-stream open actually
        # negotiated (device index, host API, channel counts). The
        # Signal Path panel shows the live values; this record keeps
        # them after the stream closes so the Atmos tester can report
        # what the PCM engine did with a file.
        self._last_open_diag: Optional[dict] = None

        # Native Atmos pipeline. `_native` is non-None while the
        # current track is playing through Windows' MediaPlayer
        # instead of PyAV. The helper is kept alive between tracks
        # (MediaPlayer reopens its source) and torn down on the same
        # calls that close the PCM path.
        self._native: Optional[NativeAtmosPlayer] = None
        self._native_info: Optional[StreamInfo] = None
        self._native_duration_ms: int = 0
        self._native_pos_ms: int = 0
        self._native_pos_at: float = 0.0
        self._native_playback_state: Optional[str] = None
        self._native_decoders: list[str] = []
        self._native_device: str = ""
        # Spatial Sound state of the render endpoint, as reported by the
        # helper: the active spatial format slug ("" = spatial off) and
        # whether the endpoint supports spatial audio at all. Refreshed
        # on every helper poll so flipping Spatial Sound mid-track moves
        # the now-playing badge instead of waiting for the next track.
        self._native_spatial_format: str = ""
        self._native_spatial_supported: bool = False
        self._native_monitor: Optional[threading.Thread] = None
        self._native_monitor_stop: Optional[threading.Event] = None
        # Loudness normalization for the native path. MediaPlayer has
        # no software DSP stage, so the offset rides on its volume
        # knob (see winrt_player.ps1). `_native_gain_source` records
        # where the number came from: "tags" (local ReplayGain tags),
        # "measured" (EBU R128 scan of the file), "pending" (scan
        # still running), or "off".
        self._native_gain_db: float = 0.0
        self._native_gain_applied_db: float = 0.0
        self._native_gain_source: str = "off"
        self._native_loudness_lufs: Optional[float] = None
        self._native_gain_lock = threading.Lock()

        # Streamed Atmos tracks, buffered as one local file per track so
        # the native pipeline can open them (see _stream_atmos_candidate).
        # The files live in user_data_dir and outlast the process, so a
        # replay (or the next app launch) skips the download; the
        # in-memory records are just the fast lookup for them.
        # `_native_stream_failures` holds the retry-after time for tracks
        # whose buffer didn't finish, so a slow asset isn't re-downloaded
        # on every click.
        self._native_stream_files: dict[str, _BufferedAtmos] = {}
        self._native_stream_failures: dict[str, float] = {}
        self._native_stream_lock = threading.Lock()
        self._native_stream_dir: Optional[str] = None
        self._atmos_cache_swept = False
        # The streamed Atmos track currently being served to MediaPlayer
        # as DASH while it downloads (see _start_dash_atmos), and the
        # look-ahead download preload() published for the queue's next
        # track, which a click adopts as-is (see _preload_atmos_stream).
        self._dash_atmos: Optional[_DashAtmos] = None
        self._dash_preloads: dict[str, _DashAtmos] = {}
        self._dash_lock = threading.Lock()
        # A spawn-idle helper holding the preloaded track's source,
        # opened paused and silent. The look-ahead pays the helper
        # spawn and the first open; the click only sets the level and
        # presses play. See _warm_native_helper.
        self._native_warm: Optional[NativeAtmosPlayer] = None
        self._native_warm_url: Optional[str] = None
        self._native_warm_reply: dict = {}
        self._native_warm_at = 0.0
        self._native_warm_lock = threading.Lock()
        # Bumped by every load()/play_track() click so an in-progress
        # buffer can abandon the download the moment the user picks
        # something else.
        self._load_seq = 0
        # The track the newest load() is for. Paired with `_load_seq`
        # to tell "a newer click is queued" apart from "this same
        # track was requested again" (double click, restore).
        self._load_target_id: Optional[str] = None

        # Preloaded next track. Populated by `preload()` ~15s before
        # the current track ends; consumed in-place by the audio
        # callback at end-of-track.
        self._preload: Optional[_Preload] = None
        # Bumped whenever a preload being built must not be installed:
        # an explicit drop (stop, a different next track) or a load.
        # preload() builds outside _pipeline_lock and only installs if
        # this hasn't moved since it started.
        self._preload_gen = 0

        # Set by _try_gapless_swap() when the audio callback
        # transitions to a new track while UPnP passthrough is active.
        # The callback can't block on HTTP calls (SetAVTransportURI),
        # so it defers to _load_locked() Path 0 which reads this flag
        # and calls start_passthrough for the new track.
        self._pending_upnp_notify: bool = False

        # Stream-manifest cache. Keyed by (track_id, quality_or_None).
        # Frontend hover / album-mount prefetch writes here so the
        # first real click is free of the playbackinfo round-trip
        # and, if bytes are warmed, of the init + first-media
        # segment fetches too. See app/audio/manifest_cache.py.
        self._manifest_cache = ManifestCache()
        # Finished streamed tracks, on disk. The bytes the decoder
        # already fetched are teed here, and a later play of the same
        # (track, quality) opens the file instead of re-downloading.
        # See app/audio/stream_cache.py.
        self._pcm_stream_cache = StreamCache()
        # One resolve per (track, quality) at a time. A hover prefetch
        # and the click it is warming overlap by design; without this
        # map they race, both issuing the same Tidal calls.
        self._resolving: dict[tuple, "_ResolveTicket"] = {}
        self._resolving_lock = threading.Lock()
        # Init + first-segment downloads started by a hover prefetch,
        # so the play that follows can take them instead of racing a
        # second copy. See _warm_bytes / _adopt_warm_bytes.
        self._warm_inflight: dict[tuple, dict[int, object]] = {}
        self._warm_lock = threading.Lock()

        # 10-band biquad EQ applied in the audio callback. One
        # instance owned for the lifetime of the player; when the
        # output stream rate changes (cross-rate bridge) we rebuild
        # it so the filter coefficients are recomputed against the
        # new sample rate. Remembered user settings (bands / preamp)
        # survive the rebuild.
        self._eq: Optional[Equalizer] = None
        # Manual parametric EQ — the user's editable band list +
        # master preamp. Remembered (not just the compiled SOS) so a
        # stream reopen recompiles coefficients at the new sample
        # rate without losing the curve.
        self._eq_parametric_bands: list[ParametricBand] = []
        self._eq_preamp: Optional[float] = None
        # AutoEQ headphone-profile mode (see
        # docs/autoeq-headphone-profiles-scope.md). Mutually
        # exclusive with `_eq_parametric_bands` — switching modes clears
        # whichever isn't active. Held here (rather than just an
        # SOS) so the stream-reopen path can recompile coefficients
        # at the new sample rate.
        self._eq_profile = None  # type: ignore[var-annotated]
        # Phase 4 A/B bypass — momentary disable that preserves
        # the active SOS / bands so flipping back is instant. The
        # audio callback reads this each frame; toggling has the
        # same ~immediate effect as a coefficient-clear without
        # the cost of rebuilding when the user toggles back on.
        self._eq_bypass: bool = False
        # Phase 5 user-tilt — bass / treble shelves + preamp
        # offset stacked on top of the active profile. Held as a
        # separate config so changing tilt without re-picking the
        # profile is one rebuild. None = no tilt set yet (treat
        # as flat); a TiltConfig with all-zero gains is also flat.
        self._eq_tilt = None  # type: ignore[var-annotated]
        # Bauer-style crossfeed for headphones — bleeds the low
        # frequencies of each channel into the opposite ear so hard-
        # panned mixes don't sound aggressive on cans. One instance
        # per output stream, rebuilt alongside the EQ on cross-rate
        # bridges. `_crossfeed_amount` (0-100 percent) is the user
        # setting we re-apply to a freshly-built filter.
        self._crossfeed: Optional[Crossfeed] = None
        self._crossfeed_amount: int = 0
        # ReplayGain loudness leveling. Single shared instance —
        # internal state is just a linear gain scalar that gets
        # recomputed whenever the active stream's tags or the user's
        # mode/preamp change. Off by default (gain_linear == 1.0).
        self._replaygain = ReplayGain()
        self._replaygain_mode: ReplayGainMode = "off"
        self._replaygain_preamp_db: float = 0.0
        self._replaygain_prevent_clipping: bool = True
        # Where the PCM path's gain came from, for `replaygain_state()`
        # and the signal-path panel. Streams carry TIDAL's tags; a
        # downloaded file carries none, so it is measured locally.
        self._pcm_gain_source: str = "none"
        self._pcm_gain_db: float = 0.0
        self._pcm_loudness_lufs: Optional[float] = None
        self._loudness_scan_path: Optional[str] = None
        self._loudness_scan_lock = threading.Lock()
        # Crossfade duration in seconds for automatic track-to-track
        # transitions (0 = off). The audio callback drives the actual
        # fade once a track nears its end and a compatible preload is
        # ready; when this is 0 that path is never entered, so playback
        # is exactly the existing gapless behaviour.
        self._crossfade_s: float = 0.0
        # In-progress fade state, owned by the audio callback. `_xfade_in`
        # is the _Preload we've claimed as the incoming track; `_xfade_pos`
        # / `_xfade_total` track the fade in samples; `_xfade_carry_in` is
        # the incoming queue's partial-chunk carry (mirrors
        # `_callback_carry` for the outgoing side).
        self._xfade_active: bool = False
        self._xfade_pos: int = 0
        self._xfade_total: int = 0
        self._xfade_in: Optional[_Preload] = None
        self._xfade_carry_in: Optional[np.ndarray] = None

        # Audio-callback diagnostics. Each pair is (count, last-print
        # time) for a different rate-limited stderr message:
        #   _cb_status: PortAudio over/underrun flags from the
        #     callback's `status` arg (driver-level glitch).
        #   _cb_starve: our decoder didn't push a chunk in time
        #     (our-side fault — slow disk / network / CPU).
        # Initialised here rather than lazy-init via getattr in the
        # callback so the attribute names live in one place and a
        # typo at the use site is a fail-fast AttributeError, not a
        # silently-zeroed counter.
        self._cb_status_count = 0
        self._cb_status_last_print = 0.0
        self._cb_starve_count = 0
        self._cb_starve_last_print = 0.0
        # Callback-jitter detection. PortAudio invokes the callback on
        # a realtime thread roughly every frames/rate seconds. When
        # something hogs the GIL/CPU (a page-navigation enrichment
        # burst, big JSON serialize) the callback is delivered late
        # even though the PCM queue is full — audible crackle that
        # never trips queue-starvation or PortAudio's own underrun
        # flag. We time the gap between consecutive audio-delivering
        # callbacks and log a missed deadline, rate-limited.
        self._cb_last_t = 0.0
        self._cb_prev_audio = False
        self._cb_jitter_count = 0
        self._cb_jitter_last_print = 0.0
        self._cb_jitter_worst_ms = 0.0
        # Cumulative (lifetime) totals of the same three glitch classes,
        # surfaced in the activity report so a stutter bug can be triaged
        # without anyone hunting for the rate-limited stderr / audio.log
        # lines. The per-second counters above reset each second; these
        # never do. Each maps to a distinct cause: status → driver /
        # exclusive-mode / device; starvation → our decode/network
        # couldn't keep up; jitter → GIL/CPU contention elsewhere.
        self._cb_status_total = 0
        self._cb_starve_total = 0
        self._cb_jitter_total = 0
        self._cb_jitter_worst_late_ms = 0.0
        # Heartbeat tick counter, only incremented when _DEBUG is on.
        # 1-per-callback log every 100 ticks confirms the callback
        # is actually running + advancing.
        self._callback_counter = 0
        # Last-bump timestamp for the audio callback's position-tick
        # nudges. The callback bumps `_seq` from here on a wall-clock
        # cadence (~5 Hz) so the SSE polling rate (4 Hz) always sees
        # a fresh seq, regardless of how often the callback itself
        # fires. The earlier "every 20 callbacks" scheme assumed an
        # 86 Hz callback rate (44.1k / 512 frames); at lower callback
        # rates — large buffers, hi-res-output configurations,
        # exclusive-mode WASAPI on devices that prefer 2048-frame
        # buffers — seq fell behind 4 Hz, the frontend's seq-dedup
        # in usePlayer kicked in, and the scrubber updated only
        # every couple of seconds instead of every 250 ms.
        self._seq_last_bump_t: float = 0.0

        # Stall watchdog. The realtime callback updates _cb_last_t
        # on every invocation (even paused/seeking — it still fires
        # to output silence), so if _cb_last_t goes stale while we
        # should be playing, the callback thread is wedged (the
        # silent-freeze class that left no log line and needed a
        # manual `sample` to diagnose). The watchdog is deliberately
        # lock-free so it can still fire while _lock is deadlocked,
        # and it dumps every thread's stack so the next freeze
        # diagnoses itself.
        self._stall_dumped = False
        # Wall-clock the watchdog uses to time how long the player has
        # sat in "loading"; reset whenever the state isn't loading. A
        # load that never produces audio (wedged segment fetch, dead
        # stream id) would otherwise pin the UI in "loading" forever.
        self._loading_since: Optional[float] = None
        # True while the load thread is blocked in the (bounded) wait
        # for a DLNA cast's per-track file to finish demuxing. That wait
        # is expected, has its own timeout (_CAST_FILE_READY_S), and for
        # long hi-res tracks routinely runs past LOAD_STALL_S. The
        # watchdog must not treat it as a wedged load, or it forces a
        # false error that the frontend's continue-playing turns into a
        # track-advance loop.
        self._awaiting_cast_file = False
        threading.Thread(
            target=self._stall_watchdog,
            name="player-stall-watchdog",
            daemon=True,
        ).start()

        # macOS audio-route listener. Subscribes to TWO CoreAudio
        # events:
        #   1. Default output device changed (headphones unplug
        #      auto-reroutes to speakers; user picks a different
        #      device in Sound prefs).
        #   2. Device list changed (a device plugged in or out,
        #      independent of whether it became the default).
        # Both feed the same handler — recovery is idempotent and
        # the player's state checks naturally dedupe.
        #
        # The dual subscription is important for "headphones
        # plugged in but macOS didn't auto-route to them" (a
        # supported Sound-pref configuration). Without the
        # device-list listener, PortAudio's enumeration stays
        # stale after that plug-in event and the picker never
        # shows the new headphones until the user restarts.
        self._audio_route_listener_unregister: Optional[
            Callable[[], None]
        ] = None
        if sys.platform == "darwin":
            try:
                from app.audio.macos_audio_devices import (
                    register_audio_route_listener,
                )
                self._audio_route_listener_unregister = (
                    register_audio_route_listener(
                        self._on_audio_route_changed
                    )
                )
                if self._audio_route_listener_unregister is not None:
                    print(
                        "[audio] subscribed to CoreAudio default-output "
                        "+ device-list changes",
                        flush=True,
                    )
            except Exception:
                log.exception(
                    "audio-route listener registration crashed"
                )

    def _on_audio_route_changed(self) -> None:
        """Fired (on a CoreAudio thread) whenever the audio routing
        topology changes — default output device flipped OR a
        device plugged in / unplugged.

        We respond with `_recover_from_device_loss` regardless of
        which event fired. The recovery is dual-purpose: when the
        selected device disappeared it falls back to default, when
        the selected device is still around it reopens on the
        original (preserving the user's manual selection AND
        refreshing PortAudio's enumeration so a newly-plugged-in
        device shows up in the picker).

        Idempotent: if the player is idle / error / ended, recovery
        short-circuits and we just need PortAudio to re-enumerate
        on the next picker open.

        Spawned on a fresh thread because we're on a CoreAudio
        internal thread here and shouldn't re-enter stream
        machinery (PortAudio reinit specifically would deadlock).
        """
        with self._lock:
            state = self._state
        if state not in ("playing", "paused", "loading"):
            # No active stream. The picker will refresh PortAudio
            # itself the next time the user opens it, so there's
            # nothing to do here.
            return
        print(
            f"[audio] audio route changed (state={state!r}); "
            "kicking recovery",
            flush=True,
        )
        threading.Thread(
            target=self._recover_from_device_loss,
            name="pcm-audio-route-changed",
            daemon=True,
        ).start()

    # --- public API -------------------------------------------------

    def subscribe(
        self, listener: Callable[[PlayerSnapshot], None]
    ) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass

        return _unsub

    def load(
        self, track_id: str, quality: Optional[str] = None
    ) -> PlayerSnapshot:
        """Resolve stream/local source, open decoder + output stream.
        Doesn't start playback; caller must `play()` next. The native
        Atmos branch obeys this too — it opens MediaPlayer paused, so
        the launch restore never makes a sound on its own.

        Serialized on `_pipeline_lock`: concurrent HTTP calls (rapid
        skip, backend auto-swap colliding with frontend play_track,
        etc.) resolve in order rather than racing into overlapping
        sounddevice streams.
        """
        with self._lock:
            # Claim this load before queueing for the pipeline. Buffering
            # a streamed Atmos track can hold the lock for a while, and a
            # newer click must be able to cut that short. getattr because
            # some unit tests build the player via __new__.
            self._load_seq = getattr(self, "_load_seq", 0) + 1
            self._load_target_id = track_id
            load_seq = self._load_seq
        with self._pipeline_lock:
            return self._load_locked(track_id, quality, load_seq)

    def _load_superseded(
        self, load_seq: Optional[int], track_id: str
    ) -> bool:
        """True when a newer load for a *different* track is queued.

        `load()` (and `play_track()`, which is the load a Next press
        actually sends) bumps `_load_seq` the moment a click arrives,
        before it waits for `_pipeline_lock`. A load still running at
        that point is dead on arrival: letting it open its source and
        call play is what made rapid Next presses audibly start each
        intermediate track. A newer load for the same track (double
        click, restore) does not supersede — it is the same work.
        """
        if load_seq is None:
            return False
        with self._lock:
            newer = self._load_seq
            target = self._load_target_id
        return newer != load_seq and target != track_id

    def _load_locked(
        self, track_id: str, quality: Optional[str], load_seq: Optional[int] = None
    ) -> PlayerSnapshot:
        self._dbg(
            f"load track={track_id} quality={quality} "
            f"current_track={self._current_track_id} "
            f"current_state={self._state} "
            f"preload={self._preload.track_id if self._preload else 'None'}"
        )
        if self._load_superseded(load_seq, track_id):
            # A newer click is already waiting for the pipeline lock.
            # Open nothing and touch nothing: the track that is
            # playing keeps playing until that newer load replaces it,
            # instead of this one flashing its audio first.
            self._dbg(f"load: superseded by a newer click; skipping {track_id}")
            return self.snapshot()
        # A look-ahead download for some other track is dead weight
        # now; one for *this* track is kept, and
        # _stream_atmos_candidate adopts it instead of downloading again.
        self._retire_dash_preloads(keep=track_id)
        # Native Atmos path: hand Atmos audio to Windows' own pipeline.
        # Local Atmos files go straight through; a streamed Atmos track
        # is buffered to one local file first, because MediaPlayer opens
        # files and TIDAL only serves segments. Checked before the
        # preload/gapless branches because a PCM preload of an Atmos
        # track would otherwise win (it decodes the same audio through
        # PyAV and skips the objects).
        candidate = self._native_candidate(
            track_id, quality, allow_stream=True, load_seq=load_seq
        )
        if candidate is not None:
            if self._load_superseded(load_seq, track_id):
                self._dbg(
                    f"load: superseded during native resolve; skipping {track_id}"
                )
                return self.snapshot()
            native_snap = self._try_load_native(track_id, *candidate)
            if native_snap is not None:
                return native_snap
        with self._lock:
            # Path 0: already audible on this track. Happens when
            # the callback's gapless swap already transitioned us
            # to the next track and the frontend's redundant
            # play_track() lands here. Don't tear anything down —
            # we're already playing what the caller asked for.
            if (
                self._current_track_id == track_id
                and self._state in ("playing", "paused")
            ):
                # Deferred UPnP passthrough notification from
                # _try_gapless_swap (audio callback can't do HTTP).
                # Start passthrough for the new track so the
                # renderer receives SetAVTransportURI before it
                # hits EOF on the previous track's buffer.
                if self._pending_upnp_notify:
                    self._pending_upnp_notify = False
                    if (
                        _upnp_manager is not None
                        and _upnp_manager.is_active()
                        and isinstance(self._source_urls, list)
                    ):
                        try:
                            _upnp_manager.start_passthrough(
                                self._source_urls,
                                prefetched=None,
                                metadata=self._current_track_meta,
                                track_id=self._current_track_id,
                            )
                        except Exception as exc:  # noqa: BLE001 - passthrough is optional
                            print(
                                f"[player] upnp start_passthrough failed "
                                f"(Path 0): {exc!r}",
                                flush=True,
                            )
                self._dbg(f"load Path 0: already playing track={track_id}")
                return self.snapshot()
            # Path 0b: a preload for this track is already buffering.
            # Adopt it without tearing down the sounddevice stream.
            # When rates + dtype match, the active OutputStream just
            # starts being fed from the preload's queue instead —
            # which is the whole point of the preload. This is the
            # path that keeps auto-advance gapless when the frontend
            # races the callback's swap.
            #
            # Requires an open OutputStream. A native Atmos track
            # plays on the OS pipeline with no sounddevice stream, so
            # adopting a preload there fed a queue nothing drained
            # while the helper kept playing the previous track — right
            # after an Atmos track, the next (ordinary) track went
            # silent. Those loads fall through to the teardown below,
            # which closes the helper first.
            pre = self._preload
            if (
                pre is not None
                and pre.track_id == track_id
                and (quality is None or pre.quality == quality)
                and self._stream is not None
                and self._state in ("playing", "paused", "loading")
            ):
                self._dbg(
                    f"load Path 0b: adopting preload track={track_id} "
                    f"(rate {pre.sample_rate}=={self._stream_sample_rate}? "
                    f"dtype {pre.sd_dtype}=={self._stream_sd_dtype}?)"
                )
                return self._adopt_preload_locked(pre)
        # Fall through: no matching preload.
        #
        # Two slow paths from here:
        #
        #  (A) **Gapless swap** when there is an active OutputStream and
        #      the new decoder can produce samples in the same rate /
        #      dtype / channel triple as the active stream. The new
        #      decoder thread runs against a fresh queue, then we
        #      atomically swap references the way `_adopt_preload_locked`
        #      does for a real preload — the OutputStream stays open
        #      across the swap, so the track change costs no stream
        #      reopen at all. (This path was first added for a USB DAC
        #      whose reopened stream failed to start with a
        #      `WdmSyncIoctl` error; that turned out to be a WASAPI
        #      start on a thread without COM, fixed in _start_stream,
        #      not the reopen itself.) When the new decoder's format
        #      doesn't fit the active stream, (A) still adopts the
        #      pipeline it built and replaces only the stream; that
        #      close + reopen can't be avoided when the format changes.
        #
        #  (B) **Full teardown + fresh open** when no stream is open yet
        #      (cold start, after a stop). This is the original slow
        #      path; it only fires when there is no stream to keep.
        self._dbg(f"load SLOW PATH: full teardown + fresh resolve for {track_id}")
        self._pending_upnp_notify = False
        # A preload still building was chosen for the track we're
        # leaving; it must not land as this new track's next.
        with self._lock:
            self._preload_gen += 1
        load_t0 = time.monotonic()

        # Snapshot the active stream's format so we can decide whether
        # path (A) is viable BEFORE doing any teardown.
        with self._lock:
            active_stream = self._stream
            active_rate = self._stream_sample_rate
            active_dtype = self._stream_sd_dtype
            active_channels = self._stream_channels

        if active_stream is not None and active_rate is not None:
            try:
                with self._lock:
                    self._transition("loading", track_id=track_id)
                synthetic = self._build_load_pipeline(
                    track_id, quality, match_rate=active_rate
                )
                t_built = time.monotonic()
            except Exception as exc:
                log.exception(
                    "failed to build pipeline for gapless swap on %s", track_id
                )
                with self._lock:
                    self._last_error = str(exc)
                    self._transition("error")
                self._emit()
                return self.snapshot()

            rate_matches = (
                synthetic.sample_rate == active_rate
                and synthetic.sd_dtype == active_dtype
                and synthetic.channels == active_channels
            )
            if self._load_superseded(load_seq, track_id):
                # The build took long enough for a newer click to be
                # queued. Discard the pipeline: swapping it in here
                # would make this intermediate track audible for the
                # fraction of a second before the newer load wins.
                self._dbg(
                    f"load: superseded during pipeline build; "
                    f"skipping {track_id}"
                )
                self._discard_pipeline(synthetic)
                return self.snapshot()
            # Path (A): adopt the pipeline we just built. When its
            # format matches the open stream, the stream stays open
            # across the swap. When it doesn't (usually a 16-bit track
            # after a 24-bit one at the mixer rate, or the reverse),
            # only the stream is replaced: this used to discard the
            # pipeline and rebuild the same one from scratch in path
            # (B), paying the resolve and the decoder init twice
            # (`teardown=~800ms` and then `decoder_init=~300ms` in the
            # load perf line, on every bit-depth change).
            outgoing_stream = None
            if not rate_matches:
                self._dbg(
                    f"load: format change "
                    f"({active_rate}/{active_dtype}/{active_channels} -> "
                    f"{synthetic.sample_rate}/{synthetic.sd_dtype}/"
                    f"{synthetic.channels}); reopening the stream"
                )
                with self._lock:
                    outgoing_stream = self._stream
                    self._stream = None
                    # Muzzle the finished_callback that abort() fires,
                    # as set_output_device does. Unmuzzled it reads the
                    # stream ending as device loss, or bridges onto the
                    # stale preload we're about to drop.
                    self._replacing_stream = True
            try:
                if outgoing_stream is not None:
                    # abort() over stop(): same reason as _teardown —
                    # stop() drains the device buffer first.
                    try:
                        outgoing_stream.abort()
                    except Exception:
                        pass
                    try:
                        outgoing_stream.close()
                    except Exception:
                        pass
                    # The callback can no longer fire, so a crossfade
                    # caught mid-window can be cancelled, as _teardown
                    # did when this case went through path (B).
                    self._abort_crossfade()
                with self._lock:
                    old_thread = self._decoder_thread
                    old_stop_flag = self._stop_flag
                    old_decoder = self._decoder
                    # Drop any stale preload (its decoder thread won't
                    # ever be adopted now that we've replaced the
                    # active pipeline with a new track).
                    stale_preload = self._preload
                    self._preload = None
                    self._swap_pipeline_to(synthetic)
                    self._source_urls = synthetic.source_urls
                    self._source_path = synthetic.source_path
                    self._paused = False
                    self._seeking = False
                    self._last_error = None
                    if outgoing_stream is not None:
                        # After the swap: _open_output_stream configures
                        # whichever decoder is current. The new
                        # decoder's thread is already running, so its
                        # output rate must not change under it; it was
                        # built against the open stream's rate (the
                        # mixer rate), which is what this asks for
                        # again, so set_target_rate is a no-op.
                        self._open_output_stream(
                            synthetic.sample_rate,
                            synthetic.channels,
                            synthetic.sd_dtype,
                        )
                    # Same contract as the original slow path: load()
                    # leaves us in "paused"; caller's play() flips to
                    # "playing" (and starts a reopened stream).
                    self._transition("paused")
            finally:
                if outgoing_stream is not None:
                    with self._lock:
                        self._replacing_stream = False

            # The old pipeline and any stale preload are detached
            # now; neither is on the audible path, so release them
            # without making this click wait for their threads.
            self._retire_pipeline(old_stop_flag, old_decoder, old_thread)
            self._retire_preload(stale_preload)

            _perf = (
                f"[perf] load track={track_id} "
                f"total={(time.monotonic() - load_t0) * 1000.0:.0f}ms "
                f"build={(t_built - load_t0) * 1000.0:.0f}ms "
                f"swap={(time.monotonic() - t_built) * 1000.0:.0f}ms "
                + (
                    "(gapless: kept stream open)"
                    if outgoing_stream is None
                    else "(format change: reopened stream)"
                )
            )
            print(_perf, file=sys.stderr, flush=True)
            # Also persist: a Finder-launched .app discards stderr,
            # so without this the slow-start breakdown is lost.
            audio_log.info(_perf)
            self._emit()
            return self.snapshot()

        # Path (B): no active stream. Full teardown + fresh open at the
        # new decoder's native rate.
        if self._load_superseded(load_seq, track_id):
            self._dbg(f"load: superseded before teardown; skipping {track_id}")
            return self.snapshot()
        # This is the Atmos -> PCM leg of a mixed queue. Park the native
        # helper rather than closing it: the Atmos track after this one
        # then only opens a source in it, instead of paying the spawn.
        self._teardown(park_native=True)
        t_teardown = time.monotonic()
        with self._lock:
            self._transition("loading", track_id=track_id)

        try:
            source_spec, duration_s, stream_info, prefetched_bytes, track_meta = (
                self._resolve_source(track_id, quality)
            )
            t_resolved = time.monotonic()

            # Start DLNA passthrough BEFORE opening the local decoder:
            # when a renderer is casting, the decoder reads the same
            # bounded file the renderer streams (so we never hold a
            # second TIDAL connection), and that file only exists once
            # start_passthrough has created it. Same condition as the
            # _build_load_pipeline path above.
            if _upnp_manager is not None and isinstance(source_spec, list):
                try:
                    if _upnp_manager.is_active():
                        _upnp_manager.start_passthrough(
                            source_spec,
                            prefetched=prefetched_bytes,
                            metadata=track_meta,
                            track_id=track_id,
                        )
                except Exception as exc:  # noqa: BLE001 - passthrough is optional
                    print(
                        f"[player] upnp start_passthrough failed "
                        f"(Path B): {exc!r}",
                        flush=True,
                    )

            cache_sink = self._pcm_stream_cache_sink(
                track_id,
                quality,
                source_spec,
                duration_s,
                stream_info,
                track_meta,
            )
            initial_source = self._open_stream_source(
                source_spec, prefetched_bytes, sink=cache_sink
            )
            decoder = Decoder(
                initial_source, surround=_is_atmos_stream_info(stream_info)
            )
            t_decoder = time.monotonic()
            if self._load_superseded(load_seq, track_id):
                # Resolving took long enough for a newer click to be
                # queued. Close this source instead of opening an
                # output stream for a track that would play for a
                # few hundred milliseconds at most.
                self._dbg(
                    f"load: superseded after resolve; skipping {track_id}"
                )
                try:
                    decoder.cancel_source()
                except Exception:
                    pass
                try:
                    decoder.close()
                except Exception:
                    pass
                return self.snapshot()
        except Exception as exc:
            # A backoff window is an expected refusal, not a bug: every
            # track click during the 60 s after a Tidal 429 lands here,
            # and a full traceback per attempt buries the log without
            # saying anything the banner doesn't. One line, plus a
            # message the now-playing bar can actually show.
            from app.tidal_client import TidalBackoffError

            if isinstance(exc, TidalBackoffError):
                log.warning(
                    "playback blocked for %s: Tidal backoff, %ss left",
                    track_id,
                    f"{exc.seconds_remaining:.0f}",
                )
                message = (
                    "TIDAL is rate-limiting requests — try again in "
                    f"{int(exc.seconds_remaining) + 1}s"
                )
            else:
                log.exception("failed to resolve/open source for %s", track_id)
                message = str(exc)
            with self._lock:
                self._last_error = message
                self._transition("error")
            self._emit()
            return self.snapshot()

        with self._lock:
            self._decoder = decoder
            self._current_duration_ms = (
                int(duration_s * 1000) if duration_s else 0
            )
            self._current_stream_info = stream_info
            self._stream_sample_rate = decoder.sample_rate
            self._stream_sd_dtype = decoder.sounddevice_dtype
            self._stream_channels = decoder.channels
            if isinstance(source_spec, str):
                self._source_path = source_spec
                self._source_urls = None
            else:
                self._source_urls = list(source_spec)
                self._source_path = None
            t_before_stream = time.monotonic()
            self._open_output_stream(
                decoder.sample_rate, decoder.channels, decoder.sounddevice_dtype
            )
            t_stream_open = time.monotonic()
            self._samples_emitted = 0
            self._callback_carry = None
            self._paused = False
            self._seeking = False
            # Fresh events + queue for the new run. An old decoder
            # thread whose join() timed out might still be alive
            # holding the PREVIOUS objects as locals. If we reuse
            # those, the zombie thread would keep pushing pre-load
            # samples into the live queue and see its stop_flag
            # toggle back to "not set," so we hand the new run a
            # clean set of objects.
            self._stop_flag = threading.Event()
            self._decoder_done = threading.Event()
            self._pcm_queue = queue.Queue(maxsize=_PCM_QUEUE_MAX)
            self._last_error = None
            # Transition out of "loading" now that the decoder is up
            # and the output stream is open. We land in "paused"
            # because nothing has called .play() yet — this is the
            # "loaded but not playing" state. play_track() will move
            # us to "playing" via its play() call right after this
            # returns; load()-without-play callers (the restore-on-
            # quit path, the album-end pause-on-track-1 flow) want
            # the snapshot to report "paused" here, not a perpetual
            # "loading" the frontend can't recover from.
            self._transition("paused")

        # Re-derive ReplayGain from the new stream's tags. The
        # helper is lock-free, so it doesn't matter that we're now
        # outside the load lock; doing it synchronously keeps the
        # gain stage in lockstep with the now-playing track instead
        # of waiting for the first audio callback to notice.
        self._apply_replaygain_for(self._current_stream_info)

        self._start_decoder_thread()
        t_thread_started = time.monotonic()
        # Emit the full breakdown at the bottom so all phases are on
        # one line. Each segment in milliseconds, total for the
        # click-to-loaded path. The two phases users care about most
        # are `resolve` (Tidal API roundtrip) and `decoder_init`
        # (manifest fetch + first segment fetch + libav probe).
        # `stream_open` is the sounddevice OutputStream open and
        # `thread_start` is the producer-thread spawn cost; both
        # stable across calls so they're floor values.
        _perf = (
            f"[perf] load track={track_id} "
            f"total={(t_thread_started - load_t0) * 1000.0:.0f}ms "
            f"teardown={(t_teardown - load_t0) * 1000.0:.0f}ms "
            f"resolve={(t_resolved - t_teardown) * 1000.0:.0f}ms "
            f"decoder_init={(t_decoder - t_resolved) * 1000.0:.0f}ms "
            f"stream_open={(t_stream_open - t_before_stream) * 1000.0:.0f}ms "
            f"thread_start="
            f"{(t_thread_started - t_stream_open) * 1000.0:.0f}ms"
        )
        print(_perf, file=sys.stderr, flush=True)
        # Also persist: a Finder-launched .app discards stderr, so
        # without this the cold-start phase breakdown is unrecoverable.
        audio_log.info(_perf)
        self._emit()
        return self.snapshot()

    # --- native Atmos pipeline --------------------------------------

    def _native_candidate(
        self,
        track_id: str,
        quality: Optional[str] = None,
        *,
        allow_stream: bool = False,
        load_seq: Optional[int] = None,
    ) -> Optional[tuple[str, Optional[float], StreamInfo]]:
        """Return `(path, duration_s, info)` when this track should
        play through the Windows pipeline.

        Only Dolby Atmos content qualifies — the native path exists to
        render the JOC object layer PyAV drops, and it sacrifices EQ /
        crossfeed / gapless, which are still worth having for ordinary
        multichannel files (5.1 FLAC etc.) that the PCM engine handles
        fine. Loudness leveling is the exception: it does run there, as
        a gain on MediaPlayer's volume stage, because a quiet immersive
        render is exactly as annoying as a quiet stereo one.

        `allow_stream` additionally buffers a TIDAL Atmos asset when no
        local file exists. It stays off on the speculative paths: a
        hover prefetch must not pull tens of MB on a guess. `preload()`
        is the exception — the next queue track is a real prediction,
        so it runs the download itself while the current track plays
        (see `_preload_atmos_stream`).
        """
        local = self._native_local_candidate(track_id)
        if local is not None:
            return local
        if not allow_stream:
            return None
        return self._stream_atmos_candidate(track_id, quality, load_seq)

    def _native_available(self) -> bool:
        """Whether the native pipeline may be used at all right now."""
        if self._native_atmos_enabled is None:
            return False
        try:
            if not self._native_atmos_enabled():
                return False
        except Exception:
            return False
        if self._external_output_active:
            # Cast / DLNA owns the audio; the native helper can't be
            # silenced the way the PCM stream can, so never start it
            # behind a remote output session.
            return False
        return native_atmos.supported()

    def _native_local_candidate(
        self, track_id: str
    ) -> Optional[tuple[str, Optional[float], StreamInfo]]:
        if not self._local_lookup:
            return None
        path = self._local_lookup(track_id)
        if not path or not os.path.exists(path):
            return None
        if os.path.splitext(path)[1].lower() not in _NATIVE_ATMOS_EXTS:
            return None
        if not self._native_available():
            return None
        if not _local_file_is_atmos(path):
            return None
        info, duration_s = _probe_local_stream_info(path)
        if info is None:
            return None
        return path, duration_s, info

    def _stream_atmos_eligible(self, track_id: str, quality: Optional[str]) -> bool:
        """Whether this track would be buffered for the native pipeline.

        Says nothing about whether TIDAL actually has an Atmos asset —
        only that the native path is the one this play should take.
        """
        if (quality or "").lower() != "atmos":
            return False
        if not self._native_available():
            return False
        if self._local_lookup is not None:
            try:
                if self._local_lookup(track_id):
                    # A local file wins; _native_local_candidate has
                    # already decided what to do with it.
                    return False
            except Exception:
                pass
        return True

    def _stream_atmos_candidate(
        self, track_id: str, quality: Optional[str], load_seq: Optional[int]
    ) -> Optional[tuple[str, Optional[float], StreamInfo]]:
        """Buffer a TIDAL Dolby Atmos stream so the native pipeline can
        play it.

        The Atmos asset only exists as a DASH segment list, so the
        segments are published to MediaPlayer one at a time and the same
        bytes are joined into one file under the cache dir as they
        arrive — the next play of that track opens the file directly.
        Returns None whenever the track isn't a streamed Atmos release
        or the stream isn't ready in time — the caller then falls
        through to the PCM engine, which decodes the same asset's
        5.1/7.1 bed exactly as before.
        """
        if not self._stream_atmos_eligible(track_id, quality):
            return None
        cached = self._native_stream_hit(track_id)
        if cached is not None:
            return cached.path, cached.duration_s, cached.info
        preload = self._dash_preload_for(track_id)
        if preload is not None:
            if preload.done_failed:
                self._retire_dash_atmos(preload)
            else:
                # preload() published this track while an earlier one
                # was playing; that head start is the whole point. The
                # click costs nothing now: MediaPlayer opens the
                # manifest and most of the segments it asks for are
                # already on disk.
                return self._adopt_dash_preload(preload)
        if self._native_stream_backoff(track_id):
            return None
        try:
            # set_current_meta stays False: the metadata is read by the
            # renderer's SetAVTransportURI, and this path only ever runs
            # with no cast session active.
            spec, duration_s, info, prefetched, _meta = self._resolve_source(
                track_id, "atmos", set_current_meta=False
            )
        except Exception as exc:
            log.debug("atmos stream resolve failed for %s: %s", track_id, exc)
            self._native_stream_failed(track_id)
            return None
        if not isinstance(spec, list) or not _is_atmos_stream_info(info):
            if isinstance(spec, list):
                # No Atmos asset: _resolve_source already fell back to
                # the best lossless tier. Alias that answer under the
                # "atmos" key so the PCM path right behind us — which
                # asks for "atmos" again — serves it from cache instead
                # of walking the Atmos lookup a second time on every
                # play of a non-Atmos track.
                self._manifest_cache.store(
                    (track_id, "atmos"), list(spec), duration_s, info, _meta
                )
            return None
        if self._dash_atmos_enabled():
            # A hover prefetch may be fetching these very segments;
            # take its bytes rather than downloading them again.
            prefetched = self._adopt_warm_bytes(
                (track_id, "atmos"), prefetched
            )
            session = self._start_dash_atmos(
                track_id, spec, prefetched, duration_s, info, load_seq
            )
            if session is not None:
                # The caller hands this manifest URL to the helper; the
                # download keeps running behind it and publishes the
                # joined file to the cache, so the next play is a plain
                # open.
                return session.url, duration_s, info
            cached = self._native_stream_hit(track_id)
            if cached is not None:
                # The first segment did not arrive in time but the
                # download did finish, so the file is still worth
                # opening.
                return cached.path, cached.duration_s, cached.info
        path = self._buffer_atmos_segments(track_id, spec, prefetched, load_seq)
        if path is None:
            if load_seq is not None and self._load_seq != load_seq:
                # A newer click abandoned this download. That is not a
                # failure: a backoff here would send the track to the
                # PCM engine for the next minute if the user comes
                # back to it.
                return None
            self._native_stream_failed(track_id)
            return None
        self._native_stream_store(track_id, path, duration_s, info)
        return path, duration_s, info

    # --- buffered stream cache --------------------------------------

    def _native_stream_hit(self, track_id: str) -> Optional[_BufferedAtmos]:
        now = time.monotonic()
        with self._native_stream_lock:
            entry = self._native_stream_files.get(track_id)
            if entry is not None:
                if not os.path.exists(entry.path):
                    self._native_stream_files.pop(track_id, None)
                elif entry.expires_at > now:
                    return entry
                elif entry.path == self._source_path:
                    # Still audible through the native pipeline. Extend
                    # rather than let it expire: a replay of this track
                    # has to reuse the file, since MediaPlayer holds it
                    # open and writing a replacement would fail.
                    entry.expires_at = now + _NATIVE_STREAM_TTL_S
                    return entry
                else:
                    # Past its TTL. The file is what is worth keeping,
                    # so re-adopt it from disk rather than re-download:
                    # the sidecar still has the stream's metadata.
                    self._native_stream_files.pop(track_id, None)
        return self._adopt_cached_atmos_file(track_id)

    def _adopt_cached_atmos_file(self, track_id: str) -> Optional[_BufferedAtmos]:
        """Pick up a buffered file written earlier — by a previous
        instance of the app, or by this one before its record aged out.

        Only the sidecar is trusted for metadata; the file is trusted
        because its recorded size matches. Anything else (no sidecar,
        size mismatch, unreadable JSON) means there is no cache to
        adopt, and the caller downloads the track as usual.
        """
        path = self._atmos_cache_path(track_id)
        payload = _read_atmos_sidecar(path)
        if payload is None:
            return None
        try:
            info = StreamInfo(**payload["info"])
        except TypeError:
            # Written by a build whose StreamInfo had different fields.
            return None
        duration_s = payload.get("duration_s")
        entry = _BufferedAtmos(
            path=path,
            duration_s=(
                float(duration_s) if isinstance(duration_s, (int, float)) else None
            ),
            info=info,
            expires_at=time.monotonic() + _NATIVE_STREAM_TTL_S,
        )
        with self._native_stream_lock:
            self._native_stream_files[track_id] = entry
        log.info("atmos cache: reusing buffered track %s", track_id)
        return entry

    def _native_stream_backoff(self, track_id: str) -> bool:
        with self._native_stream_lock:
            until = self._native_stream_failures.get(track_id)
            if until is None:
                return False
            if until > time.monotonic():
                return True
            self._native_stream_failures.pop(track_id, None)
            return False

    def _native_stream_failed(self, track_id: str) -> None:
        with self._native_stream_lock:
            self._native_stream_failures[track_id] = (
                time.monotonic() + _NATIVE_STREAM_FAILURE_TTL_S
            )

    def _native_stream_store(
        self,
        track_id: str,
        path: str,
        duration_s: Optional[float],
        info: StreamInfo,
    ) -> None:
        _write_atmos_sidecar(path, duration_s, info)
        with self._native_stream_lock:
            self._native_stream_failures.pop(track_id, None)
            self._native_stream_files[track_id] = _BufferedAtmos(
                path=path,
                duration_s=duration_s,
                info=info,
                expires_at=time.monotonic() + _NATIVE_STREAM_TTL_S,
            )
        self._native_stream_prune()

    def _native_stream_prune(self) -> None:
        """Forget records whose file is gone, then hold the disk budget.

        Expiry no longer deletes anything: the file is the expensive
        part, it is still valid, and `_native_stream_hit` re-adopts it
        from its sidecar. What gets deleted is decided by age and total
        size in `_prune_atmos_cache_files` — and never the file the
        native pipeline has open.
        """
        with self._native_stream_lock:
            for tid, entry in list(self._native_stream_files.items()):
                if not os.path.exists(entry.path):
                    self._native_stream_files.pop(tid, None)
        self._prune_atmos_cache_files()

    def _prune_atmos_cache_files(self) -> None:
        """Delete buffered tracks over the age/size budget, oldest first."""
        directory = self._atmos_stream_dir()
        try:
            names = os.listdir(directory)
        except OSError:
            return
        audible = self._source_path
        now = time.time()
        files: list[tuple[float, str, int]] = []
        for name in names:
            if not name.endswith(".m4a"):
                continue
            path = os.path.join(directory, name)
            if path == audible:
                # Open in MediaPlayer for the track that's audible;
                # removing it would fail and the bytes are in use.
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            if now - st.st_mtime > _NATIVE_STREAM_DISK_MAX_AGE_S:
                _remove_atmos_cache_file(path)
                continue
            files.append((st.st_mtime, path, st.st_size))
        total = sum(size for _mtime, _path, size in files)
        if total <= _NATIVE_STREAM_DISK_MAX_BYTES:
            return
        for _mtime, path, size in sorted(files):
            if total <= _NATIVE_STREAM_DISK_MAX_BYTES:
                break
            _remove_atmos_cache_file(path)
            total -= size
            with self._native_stream_lock:
                stale = [
                    tid
                    for tid, entry in self._native_stream_files.items()
                    if entry.path == path
                ]
                for tid in stale:
                    self._native_stream_files.pop(tid, None)

    def _atmos_cache_path(self, track_id: str) -> str:
        # track_id comes from the request body, so it is not a path
        # component: keep the name a plain file name in the cache dir.
        safe = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(track_id)
        )[:64] or "track"
        return os.path.join(self._atmos_stream_dir(), f"{safe}.m4a")

    def _atmos_stream_dir(self) -> str:
        pending_sweep = False
        with self._native_stream_lock:
            if self._native_stream_dir is None:
                self._native_stream_dir = str(
                    user_data_dir() / _NATIVE_STREAM_CACHE_DIR
                )
                pending_sweep = not self._atmos_cache_swept
                self._atmos_cache_swept = True
            resolved = self._native_stream_dir
        # The cache dir is a cache: it can be deleted at any point
        # while the process is up. Recreate it on every use instead of
        # trusting the path resolved at startup, so a delete costs a
        # re-download rather than a permanently broken track.
        try:
            os.makedirs(resolved, exist_ok=True)
        except OSError:
            # Unwritable cache root: callers fall back to streaming.
            pass
        if pending_sweep:
            self._sweep_atmos_cache(resolved)
            threading.Thread(
                target=_sweep_legacy_atmos_dirs,
                name="atmos-cache-sweep",
                daemon=True,
            ).start()
        return resolved

    def _sweep_atmos_cache(self, directory: str) -> None:
        """Clean up after previous runs: half-written buffers, orphaned
        sidecars, and files past the age budget."""
        try:
            names = os.listdir(directory)
        except OSError:
            names = []
        now = time.time()
        for name in names:
            path = os.path.join(directory, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if name.startswith("dash-") and os.path.isdir(path):
                # A DASH publication outlives its writer only while
                # MediaPlayer still reads it; one this old was left by
                # a process that died, or a delete that kept hitting
                # the open source.
                if (
                    now - st.st_mtime > 3600
                    and not dash_server.is_published_directory(path)
                ):
                    shutil.rmtree(path, ignore_errors=True)
            elif name.endswith(".json.part"):
                # Sidecar writes are instant; a leftover is debris.
                _remove_atmos_cache_file(path)
            elif name.endswith(".part"):
                # A live download keeps rewriting its part file; one
                # this old was abandoned by a process that died.
                if now - st.st_mtime > 3600:
                    _remove_atmos_cache_file(path)
            elif name.endswith(".json") and not os.path.exists(path[:-5]):
                _remove_atmos_cache_file(path[:-5])
            elif name.endswith(".m4a") and not os.path.exists(f"{path}.json"):
                # A complete buffer always gets its sidecar. One
                # without could be a write that died before finishing
                # (those cannot be deleted live — MediaPlayer holds the
                # files), so give it an hour for any writer this
                # process still has, then retire it.
                if now - st.st_mtime > 3600:
                    _remove_atmos_cache_file(path)
        self._prune_atmos_cache_files()

    def _buffer_atmos_segments(
        self,
        track_id: str,
        urls: list[str],
        prefetched: Optional[dict[int, bytes]],
        load_seq: Optional[int],
    ) -> Optional[str]:
        """Concatenate the DASH segments into one fragmented MP4.

        fMP4 is what a downloaded Atmos track is, and that exact file
        shape already plays through the native pipeline, so joining the
        segments reproduces it. Written to `.part` and renamed on
        success: a half file must never reach MediaPlayer.

        One connection is opened ahead of the body being streamed —
        the same overlap the downloader uses, and deliberately no more
        (several parallel requests per track is the concurrent-stream
        pattern TIDAL's anti-abuse layer watches). The deadline and the
        load sequence are both checked between chunks so a stalled
        download can neither pin the pipeline lock against a newer
        click nor leave the user with no audio when the PCM engine
        could be playing the bed immediately.

        This is the fallback for a stream that cannot be published as
        DASH (or a run with streaming turned off), not the normal path:
        it is the only one that makes the click wait for every segment.
        """
        from app.http import SESSION

        # `_atmos_stream_dir` recreates a deleted cache dir and also
        # serves tests that inject `_native_stream_dir` directly.
        try:
            os.makedirs(self._atmos_stream_dir(), exist_ok=True)
            final = self._atmos_cache_path(track_id)
        except OSError as exc:
            log.debug("atmos cache dir unavailable for %s: %s", track_id, exc)
            return None
        # Thread-scoped so two writers for the same track (a stream and
        # a fallback racing on a cancel) can't write into each other's
        # half file.
        target = f"{final}.{threading.get_ident()}.part"
        deadline = time.monotonic() + _NATIVE_STREAM_BUFFER_DEADLINE_S
        have = prefetched or {}
        written = 0
        t0 = time.monotonic()

        def _abandoned() -> bool:
            if load_seq is not None and self._load_seq != load_seq:
                return True
            return time.monotonic() > deadline

        def _open(url: str):
            return SESSION.get(url, stream=True, timeout=60)

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atmos-buffer")
        next_resp = None
        try:
            with open(target, "wb") as fh:

                idx = 0
                # Seeds fetched by hover/album prefetch (init + first
                # media segment) are already in memory.
                while idx in have and idx < len(urls):
                    chunk = have[idx]
                    fh.write(chunk)
                    written += len(chunk)
                    idx += 1
                if idx < len(urls):
                    next_resp = pool.submit(_open, urls[idx])
                while idx < len(urls):
                    if _abandoned():
                        raise RuntimeError("atmos stream buffer abandoned")
                    resp = next_resp.result()  # type: ignore[union-attr]
                    next_resp = (
                        pool.submit(_open, urls[idx + 1])
                        if idx + 1 < len(urls)
                        else None
                    )
                    try:
                        resp.raise_for_status()
                        for piece in resp.iter_content(chunk_size=65536):
                            if _abandoned():
                                raise RuntimeError(
                                    "atmos stream buffer abandoned"
                                )
                            if not piece:
                                continue
                            fh.write(piece)
                            written += len(piece)
                    finally:
                        try:
                            resp.close()
                        except Exception:
                            pass
                    idx += 1
            os.replace(target, final)
        except Exception as exc:
            log.debug("atmos stream buffer failed for %s: %s", track_id, exc)
            try:
                os.remove(target)
            except OSError:
                pass
            return None
        finally:
            # Never wait on a pending request here: on abandonment it can
            # still be sitting on its 60s read timeout.
            pool.shutdown(wait=False, cancel_futures=True)
        _perf = (
            f"[perf] atmos stream buffer track={track_id} "
            f"segments={len(urls)} bytes={written} "
            f"elapsed={(time.monotonic() - t0) * 1000.0:.0f}ms"
        )
        print(_perf, file=sys.stderr, flush=True)
        audio_log.info(_perf)
        return final

    # --- streaming Atmos preload ------------------------------------

    def _preload_atmos_stream(
        self, track_id: str, quality: Optional[str]
    ) -> dict:
        """Publish a streamed Atmos track as DASH before it is clicked.

        The queue's next track is known ten seconds into the current
        one, so its download can be paid for early — the same look-ahead
        Apple Music and Spotify buy by buffering the next queue item.
        It is the *same* session the click will own, not a second copy:
        one writer per track, so the click has nothing to wait for and
        nothing to duplicate. Pre-resolving here, instead of in the
        writer thread, keeps Tidal resolution serialized the way
        ordinary preloads are (the Atmos fallback path briefly flips
        `session.config.quality`).

        Returns a preload-shaped dict. `native` mirrors `preload()`'s
        existing contract for Atmos-eligible tracks; `started` / `hit`
        say whether this call published a track or found one finished.
        """
        cached = self._native_stream_hit(track_id)
        if cached is not None:
            return {"ok": True, "native": True, "hit": True}
        if self._dash_for_track(track_id) is not None:
            # This track is already downloading and published; a second
            # writer would duplicate the whole asset.
            return {"ok": True, "native": True, "hit": False}
        if not self._dash_atmos_enabled():
            # With streaming off the only pre-staging is a whole-file
            # download, and the click has to wait for that file either
            # way — the play path does it itself.
            return {"ok": True, "native": True, "hit": False}
        if self._native_stream_backoff(track_id):
            return {"ok": True, "native": True, "hit": False}

        # One look-ahead download at a time: a preload for a different
        # track makes the old one pointless, and stacking whole-asset
        # downloads is the burst pattern TIDAL's gate reads as abuse.
        self._retire_dash_preloads(keep=track_id)
        try:
            with self._pipeline_lock:
                source_spec, duration_s, info, prefetched, track_meta = (
                    self._resolve_source(track_id, "atmos", set_current_meta=False)
                )
        except Exception as exc:
            log.debug("atmos preload resolve failed for %s: %s", track_id, exc)
            return {"ok": True, "native": True, "hit": False}
        if not isinstance(source_spec, list) or not _is_atmos_stream_info(info):
            if isinstance(source_spec, list):
                # Same fallback aliasing _stream_atmos_candidate does:
                # the release has no Atmos asset, so remember the tier
                # we did get for the PCM play that follows.
                self._manifest_cache.store(
                    (track_id, "atmos"),
                    list(source_spec),
                    duration_s,
                    info,
                    track_meta,
                )
            return {"ok": True, "native": False, "hit": False}

        try:
            session = self._publish_dash_atmos(
                track_id,
                source_spec,
                prefetched,
                duration_s,
                info,
                preload=True,
            )
        except Exception as exc:
            # Cache dir deleted mid-session or otherwise unwritable: a
            # look-ahead is best-effort, and the click will retry and
            # fall back on its own.
            log.warning(
                "atmos dash: preload publish failed for %s: %s",
                track_id,
                exc,
            )
            return {"ok": True, "native": True, "hit": False}
        self._warm_native_helper(session)
        return {"ok": True, "native": True, "hit": False, "started": True}

    # --- DASH streaming ---------------------------------------------

    def _dash_atmos_enabled(self) -> bool:
        # getattr because some unit tests build the player via __new__.
        enabled = getattr(self, "_native_atmos_dash", None)
        if enabled is None:
            return False
        try:
            return bool(enabled())
        except Exception:
            return False

    def _dash_for_url(self, url: str) -> Optional[_DashAtmos]:
        with self._dash_lock:
            session = self._dash_atmos
        if session is not None and session.url == url:
            return session
        return None

    def _dash_for_track(self, track_id: str) -> Optional[_DashAtmos]:
        with self._dash_lock:
            session = self._dash_atmos
            if session is not None and session.track_id == track_id:
                return session
            return self._dash_preloads.get(track_id)

    def _dash_preload_for(self, track_id: str) -> Optional[_DashAtmos]:
        with self._dash_lock:
            return self._dash_preloads.get(track_id)

    def _take_dash_preload(self, track_id: str) -> Optional[_DashAtmos]:
        with self._dash_lock:
            return self._dash_preloads.pop(track_id, None)

    def _dash_dir(self, track_id: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(track_id)
        )[:48] or "track"
        return os.path.join(
            self._atmos_stream_dir(), f"dash-{safe}-{uuid.uuid4().hex[:8]}"
        )

    def _publish_dash_atmos(
        self,
        track_id: str,
        urls: list[str],
        prefetched: Optional[dict[int, bytes]],
        duration_s: Optional[float],
        info: StreamInfo,
        *,
        preload: bool,
    ) -> _DashAtmos:
        """Publish a track as DASH and start writing its segments.

        `preload` distinguishes a look-ahead download from one a click
        is waiting on: a look-ahead goes in the preload slot, where a
        click can adopt it, and never retires the track that is playing
        right now.
        """
        directory = self._dash_dir(track_id)
        # os.makedirs creates intermediate directories, so this also
        # recovers a cache dir removed while the app was running.
        os.makedirs(directory, exist_ok=True)
        url, publication = dash_server.publish(
            track_id,
            directory,
            max(0, len(urls) - 1),
            duration_s=duration_s,
            bandwidth=_DASH_BANDWIDTH_BPS,
        )
        publication.fetch = lambda index: self._fetch_dash_segment(urls, index)
        session = _DashAtmos(
            track_id=track_id,
            cache_path=self._atmos_cache_path(track_id),
            directory=directory,
            url=url,
            publication=publication,
            duration_s=duration_s,
            info=info,
            primed=threading.Event(),
            done=threading.Event(),
            cancel=threading.Event(),
        )
        session.done_failed = False
        previous = None
        if preload:
            # Registered before the writer starts, so a click landing in
            # between adopts this session instead of publishing another.
            with self._dash_lock:
                previous = self._dash_preloads.get(track_id)
                self._dash_preloads[track_id] = session
        threading.Thread(
            target=self._run_dash_atmos,
            args=(session, urls, prefetched),
            name=f"atmos-dash-{track_id}",
            daemon=True,
        ).start()
        if previous is not None:
            self._retire_dash_atmos(previous)
        return session

    def _adopt_dash_preload(
        self, session: _DashAtmos
    ) -> tuple[str, Optional[float], StreamInfo]:
        """Hand a look-ahead download to the track that just clicked.

        Adoption is what keeps a click free: the segments are already
        being fetched and published, so there is no second download and
        no wait. It also moves the session out of the preload slot, so
        the load path's teardown owns it like any other audible stream.
        """
        with self._dash_lock:
            if self._dash_preloads.get(session.track_id) is session:
                self._dash_preloads.pop(session.track_id, None)
            previous = self._dash_atmos
            self._dash_atmos = session
        if previous is not None and previous is not session:
            self._retire_dash_atmos(previous)
        return session.url, session.duration_s, session.info

    # --- warm helper (look-ahead for the native pipeline) ------------

    def _warm_native_helper(self, session: _DashAtmos) -> None:
        """Open a preloaded track in an idle helper, paused and silent.

        A native click pays three things the DASH prime does not cover:
        spawning the helper process, loading MediaPlayer and the Dolby
        decoder MFT, and opening the source (measured on this machine:
        1.3 s for the spawn, another 0.4 s for the first open, 0.17 s
        for the second). The look-ahead runs seconds before the click,
        so paying them there turns the click into a volume change and
        a play — which is the difference between "next song" and
        "waited for the next song".

        Only when nothing is audible natively: a playing track owns
        its helper, and a second one would be a second output device
        client with its own volume.
        """
        if not self._native_available():
            return
        if getattr(self, "_native_warm_lock", None) is None:
            return
        with self._lock:
            if self._native is not None:
                return
        with self._native_warm_lock:
            if self._native_warm is not None and self._native_warm_url == session.url:
                return
        threading.Thread(
            target=self._run_warm_helper,
            args=(session,),
            name=f"atmos-warm-{session.track_id}",
            daemon=True,
        ).start()

    def _run_warm_helper(self, session: _DashAtmos) -> None:
        # A helper parked by the track that just ended is a free
        # process; opening this source in it beats spawning another.
        helper = self._claim_idle_helper()
        if helper is None:
            helper = NativeAtmosPlayer()
        try:
            # Silent from the start: this source is opened, never played,
            # and a stray resume must not put it on the speakers.
            helper.set_volume(0.0)
            helper.set_muted(True)
            reply = helper.load(session.url, autoplay=False, adaptive=True)
        except Exception as exc:
            log.debug("atmos warm: %s could not be opened: %s", session.track_id, exc)
            helper.close()
            return
        if session.cancel.is_set():
            # The look-ahead was superseded; the process is still good.
            if not self._park_native_helper(helper):
                helper.close()
            return
        with self._native_warm_lock:
            previous = self._native_warm
            self._native_warm = helper
            self._native_warm_url = session.url
            self._native_warm_reply = dict(reply or {})
            self._native_warm_at = time.monotonic()
        if previous is not None and previous is not helper:
            threading.Thread(target=previous.close, daemon=True).start()
        audio_log.info(
            "native atmos warm: %s open and paused, waiting for the click",
            session.track_id,
        )

    def _take_warm_helper(
        self, url: str
    ) -> tuple[Optional[NativeAtmosPlayer], dict]:
        """The idle helper holding `url`, or one that can be pointed at it.

        A URL match also hands back the open reply, so the caller skips
        `load` entirely. Any other live helper comes back without a
        reply: it is still worth taking, because opening a new source
        costs ~0.17 s against ~0.5 s for spawning a fresh process. That
        is the mixed-queue case — the helper was parked by the Atmos
        track before the PCM one, so it holds that track, not this.
        """
        lock = getattr(self, "_native_warm_lock", None)
        if lock is None:
            return None, {}
        with lock:
            helper = self._native_warm
            warm_url = self._native_warm_url
            reply = self._native_warm_reply
            opened_at = self._native_warm_at
            self._native_warm = None
            self._native_warm_url = None
            self._native_warm_reply = {}
            self._native_warm_at = 0.0
        if helper is None:
            return None, {}
        stale = time.monotonic() - opened_at > _WARM_HELPER_TTL_S
        if warm_url == url and not stale:
            return helper, dict(reply)
        if stale or not getattr(helper, "running", True):
            threading.Thread(target=helper.close, daemon=True).start()
            return None, {}
        return helper, {}

    def _claim_idle_helper(self) -> Optional[NativeAtmosPlayer]:
        """Take whatever idle helper is standing, source and all.

        Used by the look-ahead: it is about to open a new source in the
        helper anyway, so a parked one is a free process.
        """
        lock = getattr(self, "_native_warm_lock", None)
        if lock is None:
            return None
        with lock:
            helper = self._native_warm
            self._native_warm = None
            self._native_warm_url = None
            self._native_warm_reply = {}
            self._native_warm_at = 0.0
        if helper is not None and not getattr(helper, "running", True):
            try:
                helper.close()
            except Exception:
                pass
            return None
        return helper

    def _park_native_helper(self, helper: NativeAtmosPlayer) -> bool:
        """Keep a finished Atmos track's helper alive, silent and idle.

        A mixed queue goes Atmos -> PCM -> Atmos, and the PCM load
        closes the live helper (it must — nothing else can stop its
        output). Closing the process means the next Atmos track pays
        the 0.4-0.8 s spawn again; parked, that click only points
        MediaPlayer at the new source (~0.17 s). Muted and at volume 0
        too, so a stray resume can never put the old track on the
        speakers behind PCM audio.

        Returns False when nothing was parked and the caller should
        close instead — a dead helper, or a look-ahead helper already
        waiting for a track that is actually next.
        """
        if not getattr(helper, "running", False):
            return False
        lock = getattr(self, "_native_warm_lock", None)
        if lock is None:
            return False
        try:
            helper.pause()
            helper.set_muted(True)
            helper.set_volume(0.0)
        except Exception as exc:
            log.debug("native atmos park failed: %s", exc)
            return False
        with lock:
            if self._native_warm is not None:
                # A warmer helper is already aimed at the next track;
                # two idle processes would be two output clients.
                return False
            self._native_warm = helper
            self._native_warm_url = None
            self._native_warm_reply = {}
            self._native_warm_at = time.monotonic()
        audio_log.info("native atmos: helper parked for the next Atmos track")
        return True

    def _reap_idle_helper(self) -> None:
        """Close an idle helper that has outlived its adoption window.

        `_take_warm_helper` would refuse it anyway; closing it here
        bounds how long a mixed queue can leave an idle process behind
        when no Atmos track ever comes.
        """
        lock = getattr(self, "_native_warm_lock", None)
        if lock is None:
            return
        with lock:
            helper = self._native_warm
            opened_at = self._native_warm_at
            if (
                helper is None
                or time.monotonic() - opened_at <= _WARM_HELPER_TTL_S
            ):
                return
            self._native_warm = None
            self._native_warm_url = None
            self._native_warm_reply = {}
            self._native_warm_at = 0.0
        threading.Thread(target=helper.close, daemon=True).start()

    def _close_warm_helper(self) -> None:
        lock = getattr(self, "_native_warm_lock", None)
        if lock is None:
            return
        with lock:
            helper = self._native_warm
            self._native_warm = None
            self._native_warm_url = None
            self._native_warm_reply = {}
            self._native_warm_at = 0.0
        if helper is not None:
            helper.close()

    def _drop_warm_helper_for(self, url: str) -> None:
        """Close the idle helper when the stream it holds is retired."""
        lock = getattr(self, "_native_warm_lock", None)
        if lock is None:
            return
        with lock:
            if self._native_warm_url != url:
                return
            helper = self._native_warm
            self._native_warm = None
            self._native_warm_url = None
            self._native_warm_reply = {}
            self._native_warm_at = 0.0
        if helper is not None:
            threading.Thread(target=helper.close, daemon=True).start()

    def _start_dash_atmos(
        self,
        track_id: str,
        urls: list[str],
        prefetched: Optional[dict[int, bytes]],
        duration_s: Optional[float],
        info: StreamInfo,
        load_seq: Optional[int],
    ) -> Optional[_DashAtmos]:
        """Publish a downloading track as DASH and wait for it to prime.

        The whole-file path makes the click wait for 20-30 MB; this
        writes the same bytes into a DASH presentation one segment at a
        time, and MediaPlayer starts as soon as the init segment and
        the first 4 s fragment exist. A look-ahead download for the same
        track is adopted rather than started again. Returns the session
        once primed, or None after cancelling the writer — the caller
        then buffers the file the old way, which is the only fallback
        that cannot leave the user with silence.
        """
        session = self._take_dash_preload(track_id)
        if session is None:
            try:
                session = self._publish_dash_atmos(
                    track_id,
                    urls,
                    prefetched,
                    duration_s,
                    info,
                    preload=False,
                )
            except Exception as exc:
                # A filesystem or publication failure must not fail the
                # load: the caller's fallback buffers the same bytes into
                # the cache dir, and the PCM engine is behind that.
                log.warning(
                    "atmos dash: publish failed for %s: %s", track_id, exc
                )
                return None
        with self._dash_lock:
            previous = self._dash_atmos
            self._dash_atmos = session

        deadline = time.monotonic() + _DASH_PRIME_DEADLINE_S
        t0 = time.monotonic()
        while not session.primed.is_set():
            if session.done.is_set() or session.done_failed:
                break
            if time.monotonic() > deadline:
                break
            if load_seq is not None and self._load_seq != load_seq:
                break
            time.sleep(_DASH_POLL_S)
        if session.primed.is_set() and not session.done_failed:
            log.info(
                "atmos dash: primed %s in %.0f ms",
                session.directory,
                (time.monotonic() - t0) * 1000.0,
            )
            # This track replaces whatever was playing, and a look-ahead
            # download for another track has nothing left to feed.
            if previous is not None and previous is not session:
                self._retire_dash_atmos(previous)
            self._retire_dash_preloads()
            return session
        self._retire_dash_atmos(session)
        # Either way this track replaces whatever was playing before, so
        # that stream has nothing left to feed.
        if previous is not None:
            self._retire_dash_atmos(previous)
        # The writer might be between segments; the fallback opens the
        # same cache path and cannot replace a file another handle
        # still holds open on Windows.
        session.done.wait(_DASH_CANCEL_WAIT_S)
        log.info("atmos dash: gave up on %s, buffering the file", track_id)
        return None

    def _fetch_dash_segment(
        self, urls: list[str], index: int
    ) -> Optional[bytes]:
        """Pull one segment out of order, for a seek.

        MediaPlayer waits on the segment under the drop point. The
        sequential pass is at best a few seconds behind, and after a
        jump to the end of a track it is minutes behind — waiting for
        it is what made a seek appear to hang. One request here answers
        the seek, and the join reuses these same bytes when it catches
        up instead of downloading them again.
        """
        from app.http import SESSION

        if index <= 0 or index >= len(urls):
            return None
        started = time.monotonic()
        try:
            resp = SESSION.get(urls[index], stream=True, timeout=30)
            try:
                resp.raise_for_status()
                data = b"".join(
                    piece
                    for piece in resp.iter_content(chunk_size=65536)
                    if piece
                )
            finally:
                resp.close()
        except Exception as exc:
            log.debug("atmos dash: seek fetch of segment %d failed: %s", index, exc)
            return None
        log.info(
            "atmos dash: segment %d on demand in %.0f ms (%d bytes)",
            index,
            (time.monotonic() - started) * 1000.0,
            len(data),
        )
        return data

    def _run_dash_atmos(
        self,
        session: _DashAtmos,
        urls: list[str],
        prefetched: Optional[dict[int, bytes]],
    ) -> None:
        """Download one track as DASH and join it into the cache file.

        Each segment is written to the published directory and appended
        to the joined file in the same pass, so the bytes are read once.
        A seek can pull a segment ahead of this pass (`fetch`); when the
        pass reaches it, the copy already on disk is joined instead of
        being downloaded a second time.
        The join is what the buffered-track cache serves from then on —
        the next play of this track opens a plain file — and it is also
        what the loudness scan measures. One connection is opened ahead
        of the body being streamed, the same overlap the whole-file
        downloader uses and deliberately no more.
        """
        from app.http import SESSION

        publication = session.publication
        part = f"{session.cache_path}.{threading.get_ident()}.part"
        deadline = time.monotonic() + _DASH_DEADLINE_S
        have = prefetched or {}
        written = 0
        t0 = time.monotonic()
        ok = False

        def _abandoned() -> bool:
            return session.cancel.is_set() or time.monotonic() > deadline

        def _open(url: str):
            return SESSION.get(url, stream=True, timeout=60)

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atmos-dash")
        next_resp = None
        def _discard(future) -> None:
            # The connection was opened ahead for bytes an out-of-order
            # fetch has since delivered; close it instead of pulling the
            # same segment down twice.
            if future is None:
                return
            try:
                future.result().close()
            except Exception:
                pass

        try:
            with open(part, "wb") as fh:

                def _write(index: int, data: bytes) -> None:
                    nonlocal written
                    fh.write(data)
                    written += len(data)
                    if index == 0:
                        dash_server.write_init(publication, data)
                    else:
                        dash_server.write_segment(publication, index, data)
                    if index == 1:
                        session.primed.set()

                idx = 0
                # Seeds fetched by hover/album prefetch (init + first
                # media segment) are already in memory — which is why
                # a primed session usually costs no new request at all.
                while idx in have and idx < len(urls):
                    _write(idx, have[idx])
                    idx += 1
                next_resp = (
                    pool.submit(_open, urls[idx]) if idx < len(urls) else None
                )
                while idx < len(urls):
                    if _abandoned():
                        raise RuntimeError("atmos dash abandoned")
                    local = publication.local_segment(idx)
                    if local is not None:
                        # A seek fetched this one out of order; the join
                        # takes those bytes instead of downloading again.
                        _discard(next_resp)
                        next_resp = None
                        _write(idx, local)
                    else:
                        resp = (
                            next_resp.result()
                            if next_resp is not None
                            else _open(urls[idx])
                        )
                        next_resp = None
                        try:
                            resp.raise_for_status()
                            pieces: list[bytes] = []
                            for piece in resp.iter_content(chunk_size=65536):
                                if _abandoned():
                                    raise RuntimeError("atmos dash abandoned")
                                if piece:
                                    pieces.append(piece)
                            _write(idx, b"".join(pieces))
                        finally:
                            try:
                                resp.close()
                            except Exception:
                                pass
                    idx += 1
                    next_resp = (
                        pool.submit(_open, urls[idx])
                        if idx < len(urls)
                        and publication.local_segment(idx) is None
                        else None
                    )
            os.replace(part, session.cache_path)
            self._native_stream_store(
                session.track_id, session.cache_path, session.duration_s, session.info
            )
            ok = True
        except Exception as exc:
            log.debug("atmos dash write failed for %s: %s", session.track_id, exc)
            try:
                os.remove(part)
            except OSError:
                pass
        finally:
            # Never wait on a pending request here: on abandonment it
            # can still be sitting on its 60 s read timeout.
            pool.shutdown(wait=False, cancel_futures=True)
            if not ok:
                session.done_failed = True
            # Pending segment requests are released either way; on a
            # failure MediaPlayer gets 404s instead of hanging.
            dash_server.note_done(publication, failed=not ok)
            session.done.set()
        _perf = (
            f"[perf] atmos dash track={session.track_id} "
            f"segments={len(urls)} bytes={written} "
            f"elapsed={(time.monotonic() - t0) * 1000.0:.0f}ms"
        )
        print(_perf, file=sys.stderr, flush=True)
        audio_log.info(_perf)

    def _finish_dash_atmos(self, session: _DashAtmos) -> Optional[str]:
        """Wait (bounded) for a refused stream to finish downloading.

        Reached when the helper will not take the adaptive source. The
        download is still running and its bytes are the only copy, so
        the fallback inside the native path opens the joined file
        rather than downloading the track again.
        """
        session.done.wait(_DASH_FALLBACK_WAIT_S)
        if session.done_failed:
            return None
        if os.path.exists(session.cache_path):
            return session.cache_path
        return None

    def _retire_dash_atmos(self, session: _DashAtmos) -> None:
        """Stop serving a DASH track and end its download.

        Non-blocking: the writer notices `cancel` between segments and
        pending requests see `done`, so nothing here waits on the
        network. The segment directory cannot be removed while
        MediaPlayer still has the source open, so the delete is retried
        briefly on a thread and the startup sweep takes what is left.
        """
        with self._dash_lock:
            if self._dash_atmos is session:
                self._dash_atmos = None
            if self._dash_preloads.get(session.track_id) is session:
                self._dash_preloads.pop(session.track_id, None)
        session.cancel.set()
        try:
            dash_server.retire(session.publication)
        except Exception as exc:
            log.debug("atmos dash retire failed: %s", exc)
        # A helper opened on this stream has nothing left to serve.
        self._drop_warm_helper_for(session.url)
        threading.Thread(
            target=_remove_dash_dir,
            args=(session.directory,),
            name="atmos-dash-cleanup",
            daemon=True,
        ).start()

    def _retire_dash_sessions(self, keep: Optional[_DashAtmos] = None) -> None:
        """End every published stream except `keep` — the track the
        caller is about to open."""
        with self._dash_lock:
            targets = [
                session
                for session in [self._dash_atmos, *self._dash_preloads.values()]
                if session is not None and session is not keep
            ]
        for session in targets:
            self._retire_dash_atmos(session)

    def _retire_dash_preloads(self, keep: Optional[str] = None) -> None:
        """End the look-ahead downloads, all but the one for `keep`."""
        with self._dash_lock:
            targets = [
                session
                for track_id, session in self._dash_preloads.items()
                if track_id != keep
            ]
        for session in targets:
            self._retire_dash_atmos(session)

    def _measured_native_gain(self, measured: dict) -> float:
        return compute_measured_gain_db(
            measured.get("lufs"),
            measured.get("peak"),
            loudness.TARGET_LUFS,
            self._replaygain_preamp_db,
            self._replaygain_prevent_clipping,
        )

    def _native_gain_plan(
        self, info: StreamInfo, path: str
    ) -> tuple[float, Optional[float], str]:
        """Resolve the leveling offset for one native track.

        Returns `(gain_db, loudness_lufs, source)`. Tags win when the
        file carries them (TIDAL's own measurement, free). A
        downloaded Atmos file has none, so fall back to the local R128
        scan — `"pending"` while the first scan of that file runs, at
        which point the track plays flat until the scan lands.
        """
        if self._replaygain_mode == "off":
            return 0.0, None, "off"
        tags = _replaygain_tags_from(info)
        if self._replaygain_mode == "track":
            has_tag = tags.track_gain_db is not None
        else:
            has_tag = tags.album_gain_db is not None
        if has_tag:
            return (
                compute_replaygain_db(
                    tags,
                    self._replaygain_mode,
                    self._replaygain_preamp_db,
                    self._replaygain_prevent_clipping,
                ),
                None,
                "tags",
            )
        if path.startswith(("http://", "https://")):
            # A DASH manifest: the joined file does not exist yet, and a
            # scan would measure however many segments have landed. The
            # writer measures the finished file (see _start_loudness_scan).
            return 0.0, None, "pending"
        measured = loudness.cached(path)
        if measured is None:
            return 0.0, None, "pending"
        lufs = measured.get("lufs")
        if not isinstance(lufs, (int, float)):
            return 0.0, None, "none"
        return self._measured_native_gain(measured), float(lufs), "measured"

    def _push_native_gain(self, helper: NativeAtmosPlayer, info: StreamInfo) -> None:
        """Re-resolve and apply the offset to a live helper. Called
        when the user changes the mode/preamp mid-track — the native
        path has no per-callback hook to pick the change up later."""
        path = self._source_path
        if not path:
            return
        gain_db, lufs, source = self._native_gain_plan(info, path)
        if source == "pending":
            # A scan is already running and will push its own result.
            return
        try:
            reply = helper.set_gain_db(gain_db)
        except Exception as exc:
            log.warning("native atmos gain failed: %s", exc)
            return
        with self._lock:
            if self._native is not helper:
                return
            self._native_gain_db = gain_db
            self._native_gain_applied_db = float(
                reply.get("gain_applied_db") or 0.0
            )
            self._native_gain_source = source
            self._native_loudness_lufs = lufs

    def _start_loudness_scan(
        self,
        helper: NativeAtmosPlayer,
        path: str,
        dash: Optional[_DashAtmos] = None,
    ) -> None:
        # play() starts this whenever the gain is still pending, so a
        # pause/resume mid-scan must not launch a second full decode.
        key = dash.cache_path if dash is not None else path
        with self._loudness_scan_lock:
            if self._loudness_scan_path == key:
                return
            self._loudness_scan_path = key

        def _run() -> None:
            try:
                target = path
                if dash is not None:
                    # A URL names a download in progress: the scan has to
                    # wait for the joined file, or it measures a truncated
                    # track and levels the next play off a bad number. A
                    # cancelled download is a partial file, which is no
                    # measurement at all.
                    if not dash.done.wait(_DASH_DEADLINE_S):
                        return
                    if dash.done_failed:
                        return
                    target = dash.cache_path
                self._scan_native_loudness(helper, target)
            finally:
                with self._loudness_scan_lock:
                    if self._loudness_scan_path == key:
                        self._loudness_scan_path = None

        thread = threading.Thread(
            target=_run,
            name="native-loudness-scan",
            daemon=True,
        )
        thread.start()

    def _scan_native_loudness(
        self, helper: NativeAtmosPlayer, path: str
    ) -> None:
        """Measure a missing-tag file and apply the result to the
        track that is currently playing it. The scan is a full decode,
        so it runs off the playback path; if the user has moved on by
        the time it finishes, the result is still cached and the next
        play of that file starts level."""
        measured = loudness.analyze(path)
        lufs = measured.get("lufs") if measured else None
        with self._lock:
            if self._native is not helper:
                return
            if lufs is None:
                self._native_gain_source = "none"
                return
            gain_db = self._measured_native_gain(measured)
        try:
            reply = helper.set_gain_db(gain_db)
        except Exception as exc:
            log.warning("native atmos loudness gain failed: %s", exc)
            return
        applied = float(reply.get("gain_applied_db") or 0.0)
        with self._lock:
            if self._native is not helper:
                return
            self._native_gain_db = gain_db
            self._native_gain_applied_db = applied
            self._native_gain_source = "measured"
            self._native_loudness_lufs = float(lufs)
            volume = self._volume
        audio_log.info(
            "native atmos loudness: %s measured %.1f LUFS, "
            "target %.1f, gain %+.2f dB (applied %+.2f dB at volume %d)",
            path,
            float(lufs),
            loudness.TARGET_LUFS,
            gain_db,
            applied,
            volume,
        )
        self._emit()

    def _native_open(
        self, helper: NativeAtmosPlayer, path: str
    ) -> tuple[dict, str]:
        """Open `path` in the helper, resolving a DASH presentation.

        `path` is either a plain file or the manifest URL of a track
        whose segments are still downloading. The helper builds an
        adaptive source from the manifest, which MediaPlayer streams
        segment by segment — the only source kind it starts before the
        whole asset exists. If it refuses, the writer is still running,
        so wait for the joined file and open that rather than
        downloading the track again. Returns the reply and the path
        that actually opened.
        """
        session = self._dash_for_url(path)
        if session is None:
            return helper.load(path, autoplay=False), path
        try:
            return helper.load(path, autoplay=False, adaptive=True), path
        except Exception as exc:
            log.warning(
                "native atmos dash open failed for %s (%s); "
                "waiting for the finished file",
                session.track_id,
                exc,
            )
            finished = self._finish_dash_atmos(session)
            if finished is None:
                raise
            return helper.load(finished, autoplay=False), finished

    def _try_load_native(
        self,
        track_id: str,
        path: str,
        duration_s: Optional[float],
        info: StreamInfo,
    ) -> Optional[PlayerSnapshot]:
        """Open `path` in the native pipeline, paused. None = failed,
        caller falls through to the PCM engine (dead helper, unsupported
        endpoint, etc.) rather than failing the play outright."""
        with self._lock:
            if (
                self._current_track_id == track_id
                and self._native is not None
                and self._state in ("playing", "paused")
            ):
                return self.snapshot()
        t0 = time.monotonic()
        # Atmos -> Atmos is the common skip, and it is the one case
        # where the helper is already standing: keep the process and
        # let `load` swap the source instead of paying the spawn.
        # A DASH session named by `path` survives the teardown too: its
        # writer is feeding the segments MediaPlayer is about to open,
        # and retiring it here would cut the new track's own source.
        # Everything else is retired, matching the old track.
        dash = self._dash_for_url(path)
        helper = self._teardown(keep_native=True, keep_dash=dash)
        t_teardown = time.monotonic()
        with self._lock:
            self._transition("loading", track_id=track_id)

        reused = helper is not None
        warm_reply: dict = {}
        if helper is None:
            # The look-ahead may have left a helper holding this exact
            # source, opened paused: adopting it skips the spawn and the
            # first open, which together are most of a cold click.
            helper, warm_reply = self._take_warm_helper(path)
            reused = helper is not None
        if helper is None:
            helper = NativeAtmosPlayer()
        gain_reply: dict = {}
        try:
            # Inside the try: a kept helper is a live process, so any
            # failure below has to reach the close() in `except` —
            # otherwise a leveling error would leave it running.
            gain_db, loudness_lufs, gain_source = self._native_gain_plan(
                info, path
            )
            if warm_reply:
                reply, opened_path = warm_reply, path
            else:
                reply, opened_path = self._native_open(helper, path)
            t_open = time.monotonic()
            if opened_path != path:
                # MediaPlayer refused the growing stream and the
                # finished file opened instead; level against that.
                gain_db, loudness_lufs, gain_source = self._native_gain_plan(
                    info, opened_path
                )
            helper.set_volume(self._volume)
            helper.set_muted(self._muted)
            # Set before play() so the first sample is already levelled
            # — MediaPlayer would otherwise start loud and drop. A
            # helper that refuses the offset must still play the
            # track: leveling is optional, the decode is not.
            try:
                gain_reply = helper.set_gain_db(gain_db)
            except Exception as exc:
                log.warning("native atmos gain failed for %s: %s", path, exc)
            # load() opens the source and stops there: the restore path
            # calls it for the persisted track on every launch and
            # expects silence until the user presses play. Starting
            # here is also what made rapid Next presses flash each
            # intermediate track. play_track() follows with play(),
            # which starts the helper and its monitor.
        except Exception as exc:
            log.warning(
                "native Atmos playback failed for %s (%s); falling back "
                "to the PCM engine",
                path,
                exc,
            )
            audio_log.warning("native atmos fallback: %s", exc)
            helper.close()
            return None

        duration_ms = (
            int(duration_s * 1000)
            if duration_s
            else int(reply.get("dur_ms") or 0)
        )
        with self._lock:
            self._native = helper
            self._native_info = info
            self._native_duration_ms = duration_ms
            self._native_pos_ms = int(reply.get("pos_ms") or 0)
            self._native_pos_at = time.monotonic()
            self._native_playback_state = "Paused"
            self._native_decoders = list(reply.get("decoders") or [])
            self._native_device = str(reply.get("device") or "")
            self._native_spatial_format = str(reply.get("spatial_format") or "")
            self._native_spatial_supported = bool(
                reply.get("spatial_supported") or False
            )
            self._native_gain_db = gain_db
            self._native_gain_applied_db = float(
                gain_reply.get("gain_applied_db") or 0.0
            )
            self._native_gain_source = gain_source
            self._native_loudness_lufs = loudness_lufs
            self._current_duration_ms = duration_ms
            self._current_stream_info = info
            self._stream_sample_rate = None
            self._stream_sd_dtype = None
            self._stream_channels = None
            self._source_path = opened_path
            self._source_urls = None
            self._samples_emitted = 0
            self._paused = True
            self._seeking = False
            self._last_error = None
            self._transition("paused")
        audio_log.info(
            "native atmos: opened %s paused decoders=%s device=%r "
            "gain=%+.2f dB (%s)",
            path,
            self._native_decoders,
            self._native_device,
            gain_db,
            gain_source,
        )
        if dash is not None and opened_path != path:
            # The adaptive source was refused, so nothing is reading the
            # segments any more; the joined file is what the cache now
            # serves.
            self._retire_dash_atmos(dash)
        # The PCM path logs a phase breakdown for every load; the native
        # path had none, which made "is the wait the download or the
        # engine switch?" unanswerable from audio.log. `helper_open`
        # covers the PowerShell spawn plus MediaPlayer opening the
        # source — the actual switch cost.
        _perf = (
            f"[perf] load native track={track_id} "
            f"total={(time.monotonic() - t0) * 1000.0:.0f}ms "
            f"teardown={(t_teardown - t0) * 1000.0:.0f}ms "
            f"helper_open={(t_open - t_teardown) * 1000.0:.0f}ms "
            f"helper={'warm' if warm_reply else ('reused' if reused else 'spawned')}"
        )
        print(_perf, file=sys.stderr, flush=True)
        audio_log.info(_perf)
        self._emit()
        return self.snapshot()

    def _start_native_monitor(self) -> None:
        with self._lock:
            stop = self._native_monitor_stop
            helper = self._native
        if stop is not None:
            stop.set()
        if helper is None:
            return
        stop = threading.Event()
        with self._lock:
            self._native_monitor_stop = stop
        thread = threading.Thread(
            target=self._native_monitor_loop,
            args=(helper, stop),
            name="native-atmos-monitor",
            daemon=True,
        )
        self._native_monitor = thread
        thread.start()

    def _native_monitor_loop(
        self, helper: NativeAtmosPlayer, stop: threading.Event
    ) -> None:
        """Keep position/state fresh and catch end-of-track.

        MediaPlayer has no event for "played to the end" that survives
        the helper's line protocol, so poll: a state that is no longer
        Playing while the position sits at the duration means EOF.

        A state that drops to Paused mid-track is the other kind of
        stop: Windows moved or reconfigured the audio device (a
        Spatial Sound change is the common one) and MediaPlayer
        parked itself. Resume it rather than leaving a "playing"
        player that is silent.
        """
        last_resume_attempt = 0.0
        resume_attempts = 0
        # MediaPlayer takes a moment to produce its first sample. While
        # it does, the UI reads this snapshot — poll tightly so the
        # track visibly starts instead of sitting at 0:00, then fall
        # back to the cheap cadence for the rest of the track.
        started = time.monotonic()
        while not stop.wait(
            0.15 if time.monotonic() - started < 2.0 else 0.5
        ):
            with self._lock:
                if self._native is not helper:
                    return
                player_state = self._state
            if player_state not in ("playing", "paused"):
                return
            try:
                reply = helper.state()
            except Exception as exc:
                with self._lock:
                    if self._native is not helper:
                        return
                    self._last_error = str(exc)
                    self._transition("error")
                audio_log.warning("native atmos helper lost: %s", exc)
                self._emit()
                return

            pos_ms = int(reply.get("pos_ms") or 0)
            dur_ms = int(reply.get("dur_ms") or 0)
            raw_state = str(reply.get("state") or "")
            playing = raw_state == "Playing"
            ended = False
            needs_resume = False
            with self._lock:
                if self._native is not helper:
                    return
                self._native_pos_ms = pos_ms
                self._native_pos_at = time.monotonic()
                self._native_playback_state = raw_state
                if dur_ms:
                    self._native_duration_ms = dur_ms
                    if dur_ms > self._current_duration_ms:
                        self._current_duration_ms = dur_ms
                self._native_decoders = list(reply.get("decoders") or [])
                device = str(reply.get("device") or "")
                if device:
                    self._native_device = device
                self._native_spatial_format = str(reply.get("spatial_format") or "")
                self._native_spatial_supported = bool(
                    reply.get("spatial_supported") or False
                )
                if (
                    not playing
                    and dur_ms > 0
                    and pos_ms >= dur_ms - 400
                    and self._state == "playing"
                ):
                    self._paused = False
                    self._transition("ended")
                    ended = True
                elif (
                    raw_state == "Paused"
                    and self._state == "playing"
                    and pos_ms < dur_ms - 400
                ):
                    needs_resume = True
            if ended:
                self._emit()
                return
            if playing:
                resume_attempts = 0
            elif needs_resume:
                now = time.monotonic()
                if now - last_resume_attempt >= 3.0 and resume_attempts < 20:
                    last_resume_attempt = now
                    resume_attempts += 1
                    try:
                        helper.play()
                        audio_log.info(
                            "native atmos: MediaPlayer dropped to Paused "
                            "at %.1fs; resumed",
                            pos_ms / 1000.0,
                        )
                    except Exception as exc:
                        audio_log.warning(
                            "native atmos: resume after drop-out failed: %s",
                            exc,
                        )

    def native_atmos_status(self) -> Optional[dict]:
        """Diagnostics for the active native stream (None when the
        current pipeline is PCM)."""
        with self._lock:
            if self._native is None:
                return None
            return {
                "path": self._source_path,
                "state": self._native_playback_state,
                "device": self._native_device,
                "decoders": list(self._native_decoders),
                "spatial_format": self._native_spatial_format,
                "spatial_supported": self._native_spatial_supported,
                "duration_ms": self._native_duration_ms,
                "gain_db": self._native_gain_db,
                "gain_applied_db": self._native_gain_applied_db,
                "gain_source": self._native_gain_source,
                "loudness_lufs": self._native_loudness_lufs,
            }

    def last_open_diagnostics(self) -> Optional[dict]:
        with self._lock:
            diag = self._last_open_diag
            return dict(diag) if diag else None

    def _clear_native_locked(
        self,
    ) -> tuple[Optional[NativeAtmosPlayer], Optional[threading.Event]]:
        """Detach the native pipeline. Returns the helper + monitor
        stop event so the caller can dispose of them outside `_lock`
        (closing the helper waits on a process handshake)."""
        with self._lock:
            helper = self._native
            stop = self._native_monitor_stop
            self._native = None
            self._native_monitor = None
            self._native_monitor_stop = None
            self._native_info = None
            self._native_duration_ms = 0
            self._native_pos_ms = 0
            self._native_pos_at = 0.0
            self._native_playback_state = None
            self._native_decoders = []
            self._native_device = ""
            self._native_spatial_format = ""
            self._native_spatial_supported = False
            self._native_gain_db = 0.0
            self._native_gain_applied_db = 0.0
            self._native_gain_source = "off"
            self._native_loudness_lufs = None
            return helper, stop

    def play_track(
        self, track_id: str, quality: Optional[str] = None
    ) -> PlayerSnapshot:
        """Combined load + play. The pipeline lock keeps load +
        play atomic relative to other state changes, so a
        concurrent stop / seek can't land in between the two
        phases.

        A pause arriving mid-load is honoured rather than discarded.
        pause() accepts input in the "loading" state and takes only
        `_lock`, so it runs freely while the resolve holds
        `_pipeline_lock` — and a cold resolve is bounded at
        `_RESOLVE_TIMEOUT_S`, which is a long time to be leaning on a
        button. Both _load_locked and play() clear `_paused` on the
        way up, so the request has to be carried across the load as a
        counter. When it moved, we stop at "loaded but not playing" —
        the same state a load()-without-play caller lands in — and the
        next play() starts the stream."""
        # Claim this click BEFORE queueing for `_pipeline_lock`, exactly
        # like load(). A rapid Next press produces one play_track per
        # track; the requests arrive together, so the ones still
        # waiting behind an in-flight load must be able to mark that
        # load dead at its next checkpoint. Bumping inside the lock
        # (as this used to, via load()) only ever happened after the
        # earlier click had finished loading and started playing its
        # intermediate track.
        with self._lock:
            self._load_seq = getattr(self, "_load_seq", 0) + 1
            self._load_target_id = track_id
            load_seq = self._load_seq
        # Sample the pause counter out here too: a pause that lands
        # while this click waits for the pipeline lock must still be
        # honoured, not overwritten when play() runs.
        pause_requests_before = self._pause_requests
        with self._pipeline_lock:
            snap = self._load_locked(track_id, quality, load_seq)
            if snap.state == "error":
                return snap
            if self._pause_requests != pause_requests_before:
                return snap
            if self._load_superseded(load_seq, track_id):
                # A newer click is queued behind this play_track. Do
                # not press play: the load either skipped its audio or
                # left the native helper holding the source paused for
                # the newer load to take over.
                self._dbg(
                    f"play_track: superseded; not starting {track_id}"
                )
                return snap
            return self.play()

    def play(self) -> PlayerSnapshot:
        with self._lock:
            native = self._native
            gain_source = self._native_gain_source
            source_path = self._source_path
        if native is not None:
            try:
                native.play()
            except Exception as exc:
                log.exception("native atmos play failed")
                with self._lock:
                    self._last_error = str(exc)
                    self._transition("error")
                self._emit()
                return self.snapshot()
            with self._lock:
                self._paused = False
                self._native_playback_state = "Playing"
                self._native_pos_at = time.monotonic()
                self._transition("playing")
                decoders = list(self._native_decoders)
                device = self._native_device
                gain_db = self._native_gain_db
            # load() opens a native source paused, so the launch restore
            # stays silent; the monitor (position + end-of-track) and
            # the loudness scan only have work once it is audible.
            self._start_native_monitor()
            if gain_source == "pending" and source_path:
                self._start_loudness_scan(
                    native, source_path, self._dash_for_url(source_path)
                )
            audio_log.info(
                "native atmos: playing %s decoders=%s device=%r "
                "gain=%+.2f dB (%s)",
                source_path,
                decoders,
                device,
                gain_db,
                gain_source,
            )
            self._emit()
            return self.snapshot()
        with self._lock:
            stream = self._stream
            if stream is None:
                # Bounded-DLNA renderer-clock mode: the muted local stream
                # is closed once its decode reaches EOF, but the renderer
                # is the audio source and keeps playing. Resume is then a
                # state transition, not a stream.start() — without this the
                # backend reported 'paused' while the renderer played, and
                # the UI clock froze.
                if _upnp_manager is None or not _upnp_manager.is_active():
                    return self.snapshot()
                self._paused = False
                self._transition("playing")
            else:
                self._paused = False
                try:
                    self._ensure_stream_started()
                except Exception as exc:
                    log.exception("stream.start failed")
                    self._last_error = str(exc)
                    self._transition("error")
                    self._emit()
                    return self.snapshot()
                self._transition("playing")
        self._emit()
        return self.snapshot()

    def pause(self) -> PlayerSnapshot:
        with self._lock:
            native = self._native
            state = self._state
        if native is not None:
            if state not in ("playing", "loading"):
                return self.snapshot()
            try:
                native.pause()
            except Exception as exc:
                log.warning("native atmos pause failed: %s", exc)
            with self._lock:
                if self._native is not native:
                    return self.snapshot()
                self._pause_requests += 1
                self._paused = True
                self._native_playback_state = "Paused"
                self._transition("paused")
            self._emit()
            return self.snapshot()
        with self._lock:
            if self._state not in ("playing", "loading"):
                return self.snapshot()
            self._pause_requests += 1
            self._paused = True
            self._transition("paused")
        self._emit()
        return self.snapshot()

    def resume(self) -> PlayerSnapshot:
        return self.play()

    def stop(self) -> PlayerSnapshot:
        # Explicit stop: drop any next track pre-staged on the renderer
        # so it can't auto-advance after the user stopped playback.
        if _upnp_manager is not None:
            try:
                _upnp_manager.invalidate_next_track()
            except Exception as exc:
                print(
                    f"[player] upnp invalidate_next_track on stop failed: "
                    f"{exc!r}",
                    flush=True,
                )
        with self._pipeline_lock:
            self._teardown()
            with self._lock:
                self._last_error = None
            self._emit()
            return self.snapshot()

    def seek(self, fraction: float) -> PlayerSnapshot:
        """Seek to `fraction` (0..1) of the track duration.

        libav's DASH/fragmented-MP4 seek doesn't work without an
        MPD-derived index, so we can't use `container.seek()` for
        Tidal streams. Instead we rebuild the decoder with a new
        `SegmentReader` that starts at the nearest media segment
        to `target_s`. Seek accuracy is segment-granular (~3-4s
        per segment at hi-res); close enough for scrub UX.

        Local files get a precise seek via `container.seek()` on
        the newly-opened container.
        """
        with self._pipeline_lock:
            fraction = max(0.0, min(1.0, float(fraction)))
            with self._lock:
                native = self._native
                duration_ms = self._current_duration_ms
                sample_rate = self._stream_sample_rate or 44100
                if duration_ms <= 0:
                    return self.snapshot()
                target_s = (duration_ms / 1000.0) * fraction
                target_ms = int(duration_ms * fraction)
            if native is not None:
                try:
                    native.seek_ms(target_ms)
                except Exception as exc:
                    log.warning("native atmos seek failed: %s", exc)
                    return self.snapshot()
                with self._lock:
                    if self._native is native:
                        self._native_pos_ms = target_ms
                        self._native_pos_at = time.monotonic()
                        self._seq += 1
                self._emit()
                return self.snapshot()
            self._dbg(
                f"seek ENTER fraction={fraction} target_s={target_s:.2f} "
                f"duration_ms={duration_ms} sample_rate={sample_rate} "
                f"samples_emitted_before={self._samples_emitted}"
            )

            # Suppress CallbackStop during the decoder swap. Without
            # this, the brief window where the queue is empty AND the
            # old decoder has exited would let the callback raise
            # CallbackStop and end the stream mid-seek.
            with self._lock:
                self._seeking = True
            # Seeking inside the crossfade window invalidates the fade —
            # the position the user jumped to bears no relation to the
            # in-progress transition. `_seeking` now silences the callback,
            # so cancel the fade and dispose the claimed incoming preload.
            self._abort_crossfade()
            # Stop DLNA passthrough on seek so the encoder doesn't
            # keep streaming from the pre-seek position. The PCM
            # re-encode path (FlacStreamEncoder via push_pcm) takes
            # over for the remainder of this track; passthrough
            # resumes automatically when the next track loads.
            # See issue #273 for context.
            #
            # BUT: while DLNA is serving a bounded per-track file the
            # renderer is the playback clock. A desktop seek must NOT
            # tear down the renderer's source — that drops it onto the
            # ~1TB synthetic RingBuffer stream, which UAPP cannot parse
            # (contentLength=1099511627776) and it stalls/fails. Here we
            # just re-seek the local decoder; the renderer keeps playing
            # this track and advances when it consumes it.
            if _upnp_manager is not None:
                try:
                    if _upnp_manager.is_active():
                        if (
                            _upnp_manager.bounded_serving()
                            and isinstance(self._source_urls, list)
                        ):
                            # The renderer is the DLNA playback clock and
                            # only follows SetAVTransportURI, so a local
                            # decoder seek is inaudible to it. Re-announce
                            # this track trimmed to the seek offset: the
                            # scrubbed/restarted position actually reaches
                            # the renderer, and the SetAVTransportURI also
                            # clears its stale NextURI.
                            _upnp_manager.start_passthrough(
                                self._source_urls,
                                start_s=target_s,
                                track_id=self._current_track_id,
                                restart=True,
                            )
                            # The re-announce cleared the renderer's next
                            # and closed its bounded file. Drop the cast
                            # preload record too, or the frontend's
                            # re-preload short-circuits as a cache hit and
                            # the next track is never re-staged.
                            with self._lock:
                                pre = self._preload
                                if (
                                    pre is not None
                                    and getattr(pre, "cast", False) is True
                                ):
                                    self._preload = None
                        else:
                            _upnp_manager.stop_passthrough()
                            # A pre-staged next the renderer still holds
                            # is wrong after a seek; drop it.
                            _upnp_manager.invalidate_next_track()
                except Exception as exc:
                    print(f"[player] upnp seek failed: {exc!r}", flush=True)
            effective_s = target_s
            try:
                effective_s = self._restart_decoder_at(target_s)
            except Exception:
                log.exception("seek to %s failed", target_s)
            finally:
                with self._lock:
                    # Use the DECODER'S effective start, not the
                    # user's requested target, so position reporting
                    # matches the audio you actually hear. On DASH
                    # this may be up to one segment earlier than
                    # requested (~3-4s).
                    self._samples_emitted = int(effective_s * sample_rate)
                    self._seeking = False
                    self._seq += 1
            return self.snapshot()

    # --- restart-based seek -----------------------------------------

    def _restart_decoder_at(self, target_s: float) -> float:
        """Tear down the current decoder + thread; build a new
        Decoder starting at or near `target_s`; start a new thread.
        Returns the effective start offset in seconds (= target_s
        for local files, = approximate segment start for DASH —
        which may be up to one segment before target_s).
        """
        seek_t0 = time.monotonic()
        self._stop_flag.set()
        # Cancel the decoder's source BEFORE joining the thread.
        # `stop_flag` only gets checked between PCM frames; if the
        # decoder is mid-segment-fetch in SegmentReader (the common
        # case during user seeks because the queue is small) we'd
        # otherwise wait out the rest of the HTTP read — 150-300 ms
        # of "nothing happens" between click and audio. cancel_source
        # closes the in-flight Response so iter_content raises and
        # the thread exits in single-digit ms.
        old_decoder = self._decoder
        if old_decoder is not None:
            try:
                old_decoder.cancel_source()
            except Exception:
                pass
        thread = self._decoder_thread
        self._decoder_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        t_thread_joined = time.monotonic()

        self._decoder = None
        if old_decoder is not None:
            try:
                old_decoder.close()
            except Exception:
                pass
        t_old_closed = time.monotonic()

        new_source, start_offset_s = self._build_source_at(target_s)
        t_source_built = time.monotonic()
        # Seek on a surround (Atmos) stream: the replacement decoder
        # must emit the same channel count the open OutputStream is
        # configured for, or every post-seek sample would be dropped by
        # the callback's shape check.
        stream_channels = self._stream_channels or 0
        new_decoder = Decoder(new_source, surround=stream_channels > 2)
        if stream_channels > 2:
            new_decoder.set_target_channels(stream_channels)
        # Match the existing OutputStream's configuration. The stream
        # stays open across a seek; the OLD decoder was configured to
        # emit at `_stream_sample_rate` / matching dtype via the
        # set_target_rate call in _open_output_stream. The NEW
        # Decoder() defaults to source-rate / source-format output —
        # which may not match the stream when shared-mode resampling
        # is in play (e.g. 96 kHz FLAC stream open at the device's
        # 48 kHz mixer rate as float32, but the new decoder produces
        # int32 at 96 kHz). The audio_callback then writes int32
        # bytes into a buffer PortAudio reads as float32, which gets
        # reinterpreted at the wrong scale and sounds like blown-out
        # bit-crushed audio. set_target_rate flips the new decoder
        # into the same flt-at-target_rate config the old one had,
        # or no-ops in the passthrough case (exclusive mode, where
        # stream rate == source rate). A float32 stream at the source
        # rate (shared mode, see _float_output_for) needs float output
        # from the new decoder too, or its int samples would be read as
        # float32 the same way.
        if self._stream_sample_rate:
            new_decoder.set_target_rate(
                self._stream_sample_rate,
                float_output=self._stream_sd_dtype == "float32",
            )
        t_decoder_init = time.monotonic()
        # For local files the new container starts at t=0; tell it
        # to seek forward to the precise target. For DASH we already
        # opened at the right segment, so the decoder starts near
        # target_s naturally.
        if start_offset_s is None:
            new_decoder.request_seek(target_s)
            effective_s = target_s
        else:
            effective_s = start_offset_s

        with self._lock:
            self._decoder = new_decoder
            self._callback_carry = None
            # Fresh events + queue — see comment in _load_locked().
            self._stop_flag = threading.Event()
            self._decoder_done = threading.Event()
            self._pcm_queue = queue.Queue(maxsize=_PCM_QUEUE_MAX)

        self._start_decoder_thread()
        t_thread_started = time.monotonic()

        # Same shape as the load() perf line so users / devs can read
        # both with the same eyes. `thread_join` covers waiting on
        # the previous decoder to exit (capped at 2 s — long values
        # here mean a stuck producer); `source_build` is the segment-
        # picking arithmetic for DASH or path lookup for local;
        # `decoder_init` is libav opening the new source. Stream
        # stays open across a seek so there's no `stream_open`
        # phase here.
        kind = "local" if start_offset_s is None else "dash"
        _perf = (
            f"[perf] seek target_s={target_s:.2f} kind={kind} "
            f"effective_s={effective_s:.2f} "
            f"total={(t_thread_started - seek_t0) * 1000.0:.0f}ms "
            f"thread_join={(t_thread_joined - seek_t0) * 1000.0:.0f}ms "
            f"old_close="
            f"{(t_old_closed - t_thread_joined) * 1000.0:.0f}ms "
            f"source_build="
            f"{(t_source_built - t_old_closed) * 1000.0:.0f}ms "
            f"decoder_init="
            f"{(t_decoder_init - t_source_built) * 1000.0:.0f}ms "
            f"thread_start="
            f"{(t_thread_started - t_decoder_init) * 1000.0:.0f}ms"
        )
        print(_perf, file=sys.stderr, flush=True)
        audio_log.info(_perf)
        return effective_s

    def _open_stream_source(
        self,
        source_spec: Union[str, list[str]],
        prefetched: Optional[dict[int, bytes]],
        sink: Optional[object] = None,
    ) -> Union[str, SegmentReader]:
        """Source for the local decoder.

        While a DLNA renderer is casting a bounded per-track file, the
        decoder reads that same local file instead of opening a second
        TIDAL connection for the track. TIDAL drops the session when two
        different tracks are open (issue #354), and the next track's
        pre-stage only starts once this file's TIDAL reader is released,
        so the local decoder must not hold a TIDAL reader of its own.
        Falls back to the URL-backed source (local playback, no cast, or
        a pre-stage that never materialised).
        """
        if (
            _upnp_manager is not None
            and isinstance(source_spec, list)
            and _upnp_manager.is_active()
        ):
            path = None
            # This wait is expected to be long on a cast (the whole
            # track is fetched and demuxed before the decoder can read
            # it), so flag it while it runs. It is bounded by
            # _CAST_FILE_READY_S, so the stuck-loading watchdog is
            # redundant here and must not fire — a false error would
            # make the frontend advance to the next track and stall
            # again, looping.
            self._awaiting_cast_file = True
            try:
                path = _upnp_manager.current_bounded_file(
                    source_spec, timeout=_CAST_FILE_READY_S
                )
            except Exception as exc:  # noqa: BLE001 - cast is optional
                path = None
                log.debug("bounded file lookup failed: %r", exc)
            finally:
                self._awaiting_cast_file = False
            if path:
                print(
                    f"[pcm] cast: decoding from bounded file {path}",
                    flush=True,
                )
                return path
            print(
                "[pcm] cast: no bounded file for this source; using URL source",
                flush=True,
            )
        return _build_source(source_spec, prefetched=prefetched, sink=sink)

    def _build_source_at(
        self, target_s: float
    ) -> tuple[Union[str, SegmentReader], Optional[float]]:
        """Return (source, approx_start_s). approx_start_s is None
        for local files (precise seek via container.seek) and the
        segment's approximate start time for DASH.
        """
        if self._source_urls is not None:
            urls = self._source_urls
            num_media = len(urls) - 1
            duration_s = (self._current_duration_ms or 1000) / 1000.0
            if num_media == 0:
                # Single-URL source (the AAC tiers' progressive MP4):
                # there is no segment to jump to, so hand the whole
                # source back with "unknown offset" and let the decoder
                # seek through the MP4 sample table. The old 0.0 return
                # told the player the new reader already started at the
                # target — every Low-tier seek restarted from 0:00.
                return SegmentReader(urls), None
            if target_s <= 0:
                return SegmentReader(urls), 0.0
            idx = max(
                0,
                min(num_media - 1, int(target_s / duration_s * num_media)),
            )
            approx_start_s = (idx / num_media) * duration_s
            # init segment + remaining media segments from idx
            sliced = [urls[0]] + list(urls[1 + idx:])
            return SegmentReader(sliced), approx_start_s
        if self._source_path is not None:
            return self._source_path, None
        raise RuntimeError("no source to seek within")

    _DEBUG = False  # verbose stderr logging — flip on while
    # debugging. At False only [pcm] state-transition prints fire.

    def _dbg(self, msg: str) -> None:
        if PCMPlayer._DEBUG:
            print(f"[pcm] {msg}", file=sys.stderr, flush=True)

    def preload(
        self, track_id: str, quality: Optional[str] = None
    ) -> dict:
        """Pre-decode the next track into a buffer so the audio
        callback can splice it into the current OutputStream at
        end-of-current-track — sample-accurate gapless.

        A streamed Atmos track can't be spliced that way: Windows'
        pipeline opens one finished file, so this starts downloading it
        in the background instead (see `_preload_atmos_stream`).

        Idempotent for the same (track_id, quality); swapping the
        request to a different track tears down the existing
        preload first. Serialized on `_pipeline_lock` alongside
        load/stop/seek.
        """
        self._dbg(f"preload ENTER track={track_id} quality={quality}")
        # Permanent, low-noise: one line per preload request. The DLNA
        # chain (upnp set_next_track -> SetNextAVTransportURI) starts
        # here, so seeing this during the current track — or not — is
        # what tells a late pre-stage apart from a preload that never
        # fired.
        print(f"[pcm] preload request track={track_id}", flush=True)
        if self._native_candidate(track_id, quality) is not None:
            # A local Atmos file: the native pipeline opens it directly,
            # so there is nothing for the PCM preload queue to hold.
            return {"ok": True, "native": True, "hit": False}
        if self._stream_atmos_eligible(track_id, quality):
            # Same, except the asset is still on TIDAL. There is no PCM
            # buffer worth priming (MediaPlayer wants the whole file),
            # so this downloads it in the background instead — that is
            # what makes the click at track-end start with no wait.
            return self._preload_atmos_stream(track_id, quality)
        with self._pipeline_lock:
            with self._lock:
                existing = self._preload
                if (
                    existing is not None
                    and existing.track_id == track_id
                    and existing.quality == quality
                ):
                    self._dbg(f"preload already cached for track={track_id}")
                    return {"ok": True, "cached": True, "hit": True}
            # Different track (or no preload) — tear down any stale
            # one before we build the new one.
            self._drop_preload()
            with self._lock:
                gen = self._preload_gen

        # The resolve and the decoder init are network round-trips —
        # several hundred ms together. They run outside _pipeline_lock
        # so a click doesn't queue behind them: the frontend fires
        # this ten seconds into every track, and a load() arriving in
        # that window used to wait out the whole preload before it
        # could start its own. What they produce is installed below
        # only if nothing has superseded this request meanwhile.
        try:
            source_spec, duration_s, stream_info, prefetched_bytes, track_meta = (
                self._resolve_source(
                    track_id, quality, set_current_meta=False
                )
            )
        except Exception as exc:
            log.exception("preload resolve failed for %s", track_id)
            return {"ok": False, "error": str(exc)}

        # While casting, do NOT open a second TIDAL reader for the
        # next track. TIDAL closes the session when two tracks are
        # open, so the next bounded file is pre-downloaded (serialized
        # after the current track's TIDAL reader is released) and then
        # pre-staged on the renderer via SetNextAVTransportURI so it
        # advances on its own at the boundary. This is the path being
        # re-tested against UAPP 7.1.2.8 (7.1.1.4 crashed in
        # HTTPStreamProvider::cleanUp on the SetNext action). The
        # staging stays serialized with the rest of the pipeline.
        if (
            _upnp_manager is not None
            and isinstance(source_spec, list)
            and _upnp_manager.is_active()
            and _upnp_manager.is_gapless()
        ):
            with self._pipeline_lock:
                if not self._preload_still_wanted(gen, track_id):
                    return {"ok": False, "superseded": True}
                staged = False
                try:
                    staged = _upnp_manager.set_next_track(
                        source_spec,
                        prefetched_bytes,
                        metadata=track_meta,
                        track_id=track_id,
                    )
                except Exception as exc:  # noqa: BLE001 - cast is optional
                    print(
                        f"[player] upnp set_next_track failed "
                        f"(cast preload): {exc!r}",
                        flush=True,
                    )
                if staged:
                    # Metadata-only record: the renderer streams the
                    # bounded file, and the renderer-clock watchdog uses
                    # this to adopt the track the renderer auto-advanced
                    # into without opening a second TIDAL reader (TIDAL
                    # closes the session when two tracks are open).
                    with self._lock:
                        self._preload = _Preload(
                            track_id=track_id,
                            quality=quality,
                            duration_ms=(
                                int(duration_s * 1000) if duration_s else 0
                            ),
                            stream_info=stream_info,
                            source_urls=list(source_spec),
                            source_path=None,
                            track_meta=track_meta,
                            cast=True,
                        )
            return {"ok": True, "cached": False, "cast": True}

        try:
            source = _build_source(
                source_spec,
                prefetched=prefetched_bytes,
                sink=self._pcm_stream_cache_sink(
                    track_id,
                    quality,
                    source_spec,
                    duration_s,
                    stream_info,
                    track_meta,
                ),
            )
            decoder = Decoder(
                source, surround=_is_atmos_stream_info(stream_info)
            )
        except Exception as exc:
            log.exception("preload resolve failed for %s", track_id)
            return {"ok": False, "error": str(exc)}

        with self._pipeline_lock:
            if not self._preload_still_wanted(gen, track_id):
                self._dbg(f"preload superseded while building track={track_id}")
                decoder.close()
                return {"ok": False, "superseded": True}

            # Configure the preload's output rate to match the active
            # stream so the gapless splice works for any source-rate
            # combination. In shared mode that's the device's mixer
            # rate, so a preloaded 96 k track decoded against a 48 k
            # stream produces 48 k output and joins seamlessly. In
            # exclusive mode we keep the preload at its own source
            # rate; if it doesn't match the active stream, the adopt
            # path reopens the stream (same ~50 ms gap as before).
            # Read now, not before the build: the stream may have been
            # reopened in a new format while we were on the network.
            current_stream_rate = self._stream_sample_rate
            if not self._exclusive_mode and current_stream_rate:
                decoder.set_target_rate(
                    current_stream_rate,
                    float_output=self._float_output_for(False),
                )

            q: queue.Queue[Optional[np.ndarray]] = queue.Queue(
                maxsize=_PCM_QUEUE_MAX
            )
            stop_flag = threading.Event()
            done = threading.Event()

            thread = threading.Thread(
                target=PCMPlayer._decoder_loop,
                args=(decoder, q, stop_flag, done),
                name="pcm-preload",
                daemon=True,
            )
            thread.start()

            urls = source_spec if isinstance(source_spec, list) else None
            path = source_spec if isinstance(source_spec, str) else None

            self._dbg(
                f"preload READY track={track_id} "
                f"rate={decoder.output_sample_rate} dtype={decoder.sounddevice_dtype} "
                f"(current stream rate={self._stream_sample_rate} "
                f"dtype={self._stream_sd_dtype})"
            )
            pre = _Preload(
                track_id=track_id,
                quality=quality,
                duration_ms=int(duration_s * 1000) if duration_s else 0,
                stream_info=stream_info,
                source_urls=list(urls) if urls is not None else None,
                source_path=path,
                decoder=decoder,
                queue=q,
                thread=thread,
                stop_flag=stop_flag,
                done=done,
                sample_rate=decoder.output_sample_rate,
                channels=decoder.channels,
                sd_dtype=decoder.sounddevice_dtype,
                track_meta=track_meta,
            )
            with self._lock:
                self._preload = pre
            # Pre-buffer the next track's DLNA passthrough while this one
            # plays, so the renderer's track-change can promote it without
            # a demux pause (gapless). No-ops for local files and when
            # no UPnP session is active.
            if pre.source_urls is not None:
                if (
                    _upnp_manager is not None
                    and _upnp_manager.is_active()
                ):
                    try:
                        # Stages the next track on the renderer via
                        # SetNextAVTransportURI (DLNA-native gapless) and
                        # builds its bounded file, which the gapped path
                        # promotes too. Returns False and leaves the
                        # gapped advance in place when the renderer
                        # doesn't support it.
                        _upnp_manager.set_next_track(
                            pre.source_urls,
                            prefetched_bytes,
                            metadata=pre.track_meta,
                        )
                    except Exception as exc:
                        print(
                            f"[player] upnp set_next_track failed: {exc!r}",
                            flush=True,
                        )
            return {
                "ok": True,
                "cached": True,
                "sample_rate": pre.sample_rate,
                "dtype": pre.sd_dtype,
            }

    def _preload_still_wanted(self, gen: int, track_id: str) -> bool:
        """Whether a preload that started at generation `gen` may still
        be installed. Call with _pipeline_lock held.

        Not if anything bumped the generation since (a drop, a newer
        preload, a load), and not if `track_id` became the current
        track meanwhile: the user clicked the track that was being
        preloaded and load() built it itself, so installing it as the
        next track would replay it at the end of itself."""
        with self._lock:
            return gen == self._preload_gen and self._current_track_id != track_id

    def _build_load_pipeline(
        self,
        track_id: str,
        quality: Optional[str],
        match_rate: Optional[int] = None,
    ) -> "_Preload":
        """Resolve `track_id`, open a decoder, start its producer
        thread, and wrap the lot as a `_Preload` for the caller to
        adopt via `_swap_pipeline_to`.

        Mirrors `preload()`'s build steps without the `_preload` slot
        assignment — the gapless load path uses this to manufacture a
        synthetic preload so it can keep the OutputStream open across a
        track change (avoiding the WASAPI `IAudioClient::Release` race
        on slow USB audio devices). `match_rate`, if provided, asks the
        decoder to emit at that rate so the swap stays format-compatible
        with the active stream; pass `None` to keep the decoder's native
        rate.

        Raises on resolve / decoder construction failures. Caller is
        responsible for tearing down the returned pipeline if it
        decides not to adopt it.
        """
        source_spec, duration_s, stream_info, prefetched_bytes, track_meta = (
            self._resolve_source(track_id, quality)
        )
        # Start bit-perfect FLAC passthrough when a DLNA session is
        # active. The passthrough creates the bounded per-track file the
        # renderer streams; the local decoder then reads that same file
        # (see _open_stream_source) so only one TIDAL track is ever open.
        # Only when the source is a URL list (Tidal stream), not a
        # local file.
        if _upnp_manager is not None and isinstance(source_spec, list):
            try:
                if _upnp_manager.is_active():
                    _upnp_manager.start_passthrough(
                        source_spec,
                        prefetched=prefetched_bytes,
                        metadata=track_meta,
                        track_id=track_id,
                    )
            except Exception as exc:  # noqa: BLE001 - passthrough is optional
                print(
                    f"[player] upnp start_passthrough failed: {exc!r}",
                    flush=True,
                )

        source = self._open_stream_source(
            source_spec,
            prefetched_bytes,
            sink=self._pcm_stream_cache_sink(
                track_id,
                quality,
                source_spec,
                duration_s,
                stream_info,
                track_meta,
            ),
        )
        decoder = Decoder(
            source, surround=_is_atmos_stream_info(stream_info)
        )

        # Match the active stream's rate so the adopt path can take
        # the same-rate (stream-stays-open) branch. Skipped in
        # exclusive mode because we want the source's native rate
        # there — if it ends up differing from the active stream's,
        # the caller falls through to the full teardown path.
        if match_rate is not None and not self._exclusive_mode:
            decoder.set_target_rate(
                int(match_rate), float_output=self._float_output_for(False)
            )

        q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(
            maxsize=_PCM_QUEUE_MAX
        )
        stop_flag = threading.Event()
        done = threading.Event()

        thread = threading.Thread(
            target=PCMPlayer._decoder_loop,
            args=(decoder, q, stop_flag, done),
            name="pcm-decoder",
            daemon=True,
        )
        thread.start()

        urls = source_spec if isinstance(source_spec, list) else None
        path = source_spec if isinstance(source_spec, str) else None

        return _Preload(
            track_id=track_id,
            quality=quality,
            duration_ms=int(duration_s * 1000) if duration_s else 0,
            stream_info=stream_info,
            source_urls=list(urls) if urls is not None else None,
            source_path=path,
            decoder=decoder,
            queue=q,
            thread=thread,
            stop_flag=stop_flag,
            done=done,
            sample_rate=decoder.output_sample_rate,
            channels=decoder.channels,
            sd_dtype=decoder.sounddevice_dtype,
        )

    def _drop_preload(self) -> None:
        """Stop the preload thread, close its decoder, clear the slot.
        Safe to call from either a locked pipeline operation or from
        the HTTP layer — takes the pipeline lock itself (RLock, so
        nesting is fine).
        """
        with self._pipeline_lock:
            with self._lock:
                pre = self._preload
                self._preload = None
                self._preload_gen += 1
            if pre is None:
                return
            print(f"[pcm] drop preload track={pre.track_id}", flush=True)
            self._retire_preload(pre)

    def set_volume(self, volume: int) -> PlayerSnapshot:
        volume = max(0, min(100, int(volume)))
        with self._lock:
            # Force Volume wins over caller requests — keeps the
            # signal chain bit-perfect. UI disables the slider in
            # that state, but external clients can still try and
            # we just ignore them.
            if not self._force_volume:
                self._volume = volume
            self._seq += 1
            native = self._native
            effective = self._volume
        if native is not None:
            # MediaPlayer has its own volume stage; keep it in step
            # with the slider or the two would fight.
            try:
                native.set_volume(effective)
            except Exception as exc:
                log.warning("native atmos volume failed: %s", exc)
        return self.snapshot()

    def set_force_volume(self, enabled: bool) -> PlayerSnapshot:
        """Toggle Force Volume. Turning it on pins _volume to 100
        immediately so any subsequent playback is at full scale."""
        with self._lock:
            self._force_volume = bool(enabled)
            if self._force_volume:
                self._volume = 100
            self._seq += 1
            native = self._native
            effective = self._volume
        if native is not None:
            try:
                native.set_volume(effective)
            except Exception as exc:
                log.warning("native atmos volume failed: %s", exc)
        return self.snapshot()

    def set_muted(self, muted: bool) -> PlayerSnapshot:
        with self._lock:
            self._muted = bool(muted)
            self._seq += 1
            native = self._native
            muted_value = self._muted
        if native is not None:
            try:
                native.set_muted(muted_value)
            except Exception as exc:
                log.warning("native atmos mute failed: %s", exc)
        return self.snapshot()

    def get_current_source_urls(self) -> Optional[list[str]]:
        """Return the active track's segment URLs, or None if no
        track is loaded or the source is a local file.

        Used by the DLNA manager to start FLAC passthrough when a
        session is established after a track is already playing."""
        with self._lock:
            return self._source_urls

    def get_current_track_metadata(self) -> Optional[dict]:
        """Return the active track's metadata dict, or None."""
        with self._lock:
            return self._current_track_meta

    def get_preloaded_source(
        self,
    ) -> Optional[tuple[list[str], Optional[dict]]]:
        """Return the preloaded NEXT track's segment URLs and metadata,
        or None when no URL-backed preload is buffering.

        Used by the DLNA manager when a session connects: switching
        output to a renderer mid-track can land after the next track was
        already preloaded. Without staging that preload on the renderer,
        it reaches the end of the current track with no `next` and
        stops (issue #354)."""
        with self._lock:
            pre = self._preload
            if pre is None or pre.source_urls is None:
                return None
            return list(pre.source_urls), getattr(pre, "track_meta", None)

    def get_current_position_s(self) -> Optional[float]:
        """Current playback position in seconds, or None when unknown.

        Used by the DLNA manager to trim the bounded file served to the
        renderer so it begins where the desktop decoder already is. In
        DLNA the renderer is the playback clock; if the two start at
        different offsets (a resume-from-position), the decoder's EOF
        fires early and the next track's SetAVTransportURI cuts the
        tail (issue #354)."""
        with self._lock:
            if not self._stream_sample_rate or self._samples_emitted <= 0:
                return None
            return self._samples_emitted / self._stream_sample_rate

    def set_external_output_active(self, active: bool) -> None:
        """Toggle local-output silencing.

        Called by the Cast / Tidal Connect managers when
        a remote output session opens or closes. While true, the
        audio callback writes silence to the local sounddevice
        OutputStream so the user doesn't hear duplicate audio from
        Mac speakers and the remote device. Source PCM still
        flows to the remote tap before silencing, so the receiver
        gets full-amplitude audio at its own volume control.

        Idempotent. Doesn't change `_volume` or `_muted` — the
        user's manual mute / volume settings are preserved across
        a cast cycle so they don't have to re-set them when audio
        returns to local.
        """
        with self._lock:
            self._external_output_active = bool(active)
            self._seq += 1

    # --- EQ --------------------------------------------------------
    #
    # Per-track coefficients depend on sample_rate, so the
    # Equalizer instance is built / rebuilt in load() /
    # _adopt_preload_locked / _bridge_to_preload, but the user's
    # bands + preamp are remembered on the player and re-applied
    # each rebuild.

    def apply_equalizer(
        self, bands: list, preamp: Optional[float] = None
    ) -> None:
        """Apply a manual parametric band list to the active EQ.

        `bands` is a list of `ParametricBand` (or band dicts, which
        are validated/coerced here). Flat and disabled bands compile
        to no biquads; `set_sos` handles the empty-cascade cases —
        full bypass when the preamp is also unity, a preamp-only
        stage otherwise (the user's preamp must stay audible even
        when every band is flat).

        Remembered so that a stream reopen (cross-rate bridge, track
        load) rebuilds the EQ coefficients against the new sample
        rate without losing the user's curve.

        Calling this also clears any active AutoEQ profile — manual
        and profile modes are mutually exclusive in the player.
        """
        coerced = [
            b if isinstance(b, ParametricBand) else parametric_band_from_dict(b)
            for b in bands
        ]
        with self._lock:
            self._eq_parametric_bands = coerced
            self._eq_preamp = preamp
            self._eq_profile = None
            if self._eq is not None:
                sos = build_parametric_sos(coerced, self._eq.sample_rate())
                self._eq.set_sos(sos, preamp_db=preamp)

    def set_equalizer_bypass(self, bypassed: bool) -> None:
        """Toggle the A/B bypass flag without touching the active
        EQ configuration. The audio callback reads this each
        frame, so the effect is on the next callback (~10 ms).
        Used by the Phase 4 player-UI button + keyboard shortcut
        for blink-test comparison."""
        with self._lock:
            self._eq_bypass = bool(bypassed)

    def set_crossfeed_amount(self, amount_pct: int) -> None:
        """Set the Bauer crossfeed strength (0-100 percent). 0
        disables the stage entirely; non-zero values install /
        update the low-pass that bleeds bass between channels.
        Surface for the Settings page slider."""
        clamped = max(0, min(100, int(amount_pct)))
        with self._lock:
            self._crossfeed_amount = clamped
            if self._crossfeed is not None:
                if clamped > 0:
                    try:
                        self._crossfeed.set_amount(clamped)
                    except Exception:
                        log.exception("crossfeed coefficient build failed")
                        self._crossfeed.clear()
                else:
                    self._crossfeed.clear()

    def crossfeed_amount(self) -> int:
        """Current crossfeed setting in percent. 0 means bypassed."""
        with self._lock:
            return self._crossfeed_amount

    def set_crossfade(self, seconds) -> None:
        """Set the automatic-advance crossfade duration in seconds
        (clamped 0-12; 0 = off). Stored here; the fade itself is driven
        from the audio callback when a track nears its end and a
        compatible, sufficiently-buffered preload is ready. Surface for
        the Settings page slider."""
        try:
            s = float(seconds)
        except (TypeError, ValueError):
            s = 0.0
        s = max(0.0, min(12.0, s))
        with self._lock:
            self._crossfade_s = s

    def crossfade_duration(self) -> float:
        """Current crossfade duration in seconds. 0 means disabled."""
        with self._lock:
            return self._crossfade_s

    def set_replaygain(
        self,
        mode: str,
        preamp_db: float,
        prevent_clipping: bool,
    ) -> None:
        """Configure ReplayGain leveling. `mode` is "off", "track",
        or "album"; unknown modes coerce to "off". The applied gain
        is recomputed against the currently-loaded stream's tags so
        the change takes effect immediately, not on the next track.
        """
        normalized: ReplayGainMode = (
            mode if mode in REPLAYGAIN_VALID_MODES else "off"
        )  # type: ignore[assignment]
        # Single-attribute writes are GIL-atomic; we don't need the
        # player lock here. Acquiring it would be safe but creates a
        # subtle deadlock risk for swap paths that already hold it.
        self._replaygain_mode = normalized
        self._replaygain_preamp_db = float(preamp_db)
        self._replaygain_prevent_clipping = bool(prevent_clipping)
        # Re-derive gain off the active stream's tags so the change
        # is audible immediately rather than on the next track.
        self._apply_replaygain_for(self._current_stream_info)

    def replaygain_state(self) -> dict:
        """Current configuration + the resolved gain in dB applied
        right now. Surface for the Settings page so users can see
        whether the active stream actually has tags or fell back to
        flat output. On the native Atmos path the number and its
        provenance come from the helper, which is the only pipeline
        where the offset is a measured loudness rather than a tag."""
        mode = self._replaygain_mode
        preamp = self._replaygain_preamp_db
        prevent = self._replaygain_prevent_clipping
        tags = _replaygain_tags_from(self._current_stream_info)
        applied_db = compute_replaygain_db(
            tags,
            mode,
            preamp,
            prevent,
            _tag_reference_shift_db(self._current_stream_info),
        )
        has_tag = (
            tags.track_gain_db if mode == "track" else tags.album_gain_db
        ) is not None
        source = "tags" if (mode != "off" and has_tag) else "none"
        loudness_lufs: Optional[float] = None
        native = self._native
        if native is not None:
            applied_db = self._native_gain_db
            source = self._native_gain_source
            loudness_lufs = self._native_loudness_lufs
        elif mode != "off" and not has_tag:
            # PCM local file with no tags: the number comes from the
            # local measurement, or the track is still playing flat
            # because that scan hasn't finished yet.
            applied_db = self._pcm_gain_db
            source = self._pcm_gain_source
            loudness_lufs = self._pcm_loudness_lufs
        return {
            "mode": mode,
            "preamp_db": preamp,
            "prevent_clipping": prevent,
            "applied_db": applied_db,
            "applied_effective_db": (
                self._native_gain_applied_db if native is not None else applied_db
            ),
            "source": source,
            "loudness_lufs": loudness_lufs,
            "track_gain_db": tags.track_gain_db,
            "track_peak": tags.track_peak,
            "album_gain_db": tags.album_gain_db,
            "album_peak": tags.album_peak,
        }

    def _apply_replaygain_for(self, info: Optional[StreamInfo]) -> None:
        """Install the gain that corresponds to `info`'s tags + the
        current user mode/preamp/clipping settings. Lock-free —
        callers can be in audio-callback / lock-holding contexts
        without re-entrancy concerns. The ReplayGain instance has
        its own internal lock for the actual coefficient swap.

        Tags are not the only source: a downloaded file carries none
        (the downloader muxes audio only), and tags alone left such a
        track playing at its master level — a -6 LUFS pop master
        stayed ~10 dB louder than the same song in Apple Music. So a
        tag-less local file is measured locally, with the same EBU
        R128 meter the native Atmos path uses, and the offset is
        applied once the scan lands.
        """
        if info is None:
            self._replaygain.clear()
            self._pcm_gain_source = "none"
            self._pcm_gain_db = 0.0
            self._pcm_loudness_lufs = None
            return
        tags = _replaygain_tags_from(info)
        gain_db = compute_replaygain_db(
            tags,
            self._replaygain_mode,
            self._replaygain_preamp_db,
            self._replaygain_prevent_clipping,
            _tag_reference_shift_db(info),
        )
        self._replaygain.set_gain_db(gain_db)
        # The native pipeline has no sample buffer to multiply, so its
        # offset lives on the helper instead. PCM and native never run
        # together, which is what keeps this branch off the callback's
        # path: when a helper is attached, the PCM callback is dormant.
        helper = self._native
        if helper is not None:
            self._push_native_gain(helper, info)
            return

        has_tag = (
            tags.track_gain_db
            if self._replaygain_mode == "track"
            else tags.album_gain_db
        ) is not None
        self._pcm_gain_source = (
            "tags" if (self._replaygain_mode != "off" and has_tag) else "none"
        )
        self._pcm_gain_db = gain_db
        self._pcm_loudness_lufs = None
        if self._replaygain_mode == "off" or has_tag:
            return
        if info.source != "local" or not self._source_path:
            return
        path = self._source_path
        measured = loudness.cached(path)
        if measured is None:
            # First play of this file: it stays flat until the scan
            # finishes rather than guessing a level.
            self._pcm_gain_source = "pending"
            self._start_pcm_loudness_scan(path)
            return
        self._apply_measured_pcm_gain(path, measured)

    def _start_pcm_loudness_scan(self, path: str) -> None:
        """Measure one tag-less local file in the background. A play
        that lands while the scan runs just waits for its result."""
        with self._loudness_scan_lock:
            if self._loudness_scan_path == path:
                return
            self._loudness_scan_path = path
        threading.Thread(
            target=self._scan_pcm_loudness,
            args=(path,),
            name="pcm-loudness-scan",
            daemon=True,
        ).start()

    def _scan_pcm_loudness(self, path: str) -> None:
        try:
            measured = loudness.analyze(path)
        finally:
            with self._loudness_scan_lock:
                if self._loudness_scan_path == path:
                    self._loudness_scan_path = None
        if measured:
            self._apply_measured_pcm_gain(path, measured)

    def _apply_measured_pcm_gain(self, path: str, measured: dict) -> None:
        """Apply a measured offset when `path` is still what's playing.
        The measurement is asynchronous, so the queue may have moved on
        by the time it lands."""
        if self._source_path != path or self._replaygain_mode == "off":
            return
        lufs = measured.get("lufs")
        if not isinstance(lufs, (int, float)):
            self._pcm_gain_source = "none"
            return
        gain_db = compute_measured_gain_db(
            float(lufs),
            measured.get("peak"),
            loudness.TARGET_LUFS,
            self._replaygain_preamp_db,
            self._replaygain_prevent_clipping,
        )
        self._replaygain.set_gain_db(gain_db)
        self._pcm_gain_source = "measured"
        self._pcm_gain_db = gain_db
        self._pcm_loudness_lufs = float(lufs)
        try:
            log.info(
                "loudness leveling: measured %.1f LUFS -> %+.1f dB (PCM, %s)",
                float(lufs),
                gain_db,
                os.path.basename(path),
            )
        except Exception:
            pass

    def equalizer_bypass(self) -> bool:
        """Read the current bypass flag — server uses it to expose
        the value via /api/eq/state."""
        with self._lock:
            return self._eq_bypass

    def apply_equalizer_profile(self, profile) -> None:
        """Switch to AutoEQ profile mode and apply `profile`. The
        profile object's bands compile to an SOS at the player's
        current sample rate; on a stream reopen the same profile
        is recompiled at the new rate (no loss across cross-rate
        bridges). Clears any manual-mode bands — the modes are
        mutually exclusive.

        Phase 5: if a user-tilt is active, the cascade includes
        the tilt shelves + preamp offset. Picking a new profile
        keeps the tilt setting (taste preference travels with
        the user, not the headphone)."""
        from app.audio.autoeq.apply import (
            TiltConfig,
            cascade_with_tilt,
        )

        with self._lock:
            self._eq_parametric_bands = []
            self._eq_preamp = None
            self._eq_profile = profile
            tilt = self._eq_tilt or TiltConfig()
            if self._eq is not None:
                sos, preamp_db = cascade_with_tilt(
                    profile, self._eq.sample_rate(), tilt
                )
                if sos.size == 0:
                    self._eq.clear()
                else:
                    self._eq.set_sos(sos, preamp_db=preamp_db)

    def apply_equalizer_tilt(self, tilt) -> None:
        """Update the user-tilt and rebuild the active cascade if
        a profile is loaded. No-op when no profile is active —
        tilt is a stack-on-top thing, not a standalone EQ.

        `tilt` is `TiltConfig | None`; passing None clears tilt
        back to flat. The profile + sample rate stay the same;
        only the trailing tilt shelves + preamp offset change."""
        from app.audio.autoeq.apply import (
            TiltConfig,
            cascade_with_tilt,
        )

        with self._lock:
            self._eq_tilt = tilt
            effective = tilt if tilt is not None else TiltConfig()
            if self._eq_profile is None or self._eq is None:
                return
            sos, preamp_db = cascade_with_tilt(
                self._eq_profile, self._eq.sample_rate(), effective
            )
            if sos.size == 0:
                self._eq.clear()
            else:
                self._eq.set_sos(sos, preamp_db=preamp_db)

    def apply_equalizer_preset(self, preset_index: int) -> list[dict]:
        """Apply a preset by index, push its curve to the live EQ,
        and return the resolved parametric bands (as dicts) so the
        frontend can snap its editor to the preset."""
        bands = parametric_preset(preset_index)
        self.apply_equalizer(bands, preamp=None)
        return [b.to_dict() for b in bands]

    # --- Device selection -------------------------------------------

    def output_stream_state(self) -> dict:
        """Snapshot of the open OutputStream — what's actually being
        fed to the OS audio API right now. Surface for the Signal
        Path readout so audiophiles can see the realised output
        format alongside the source format. All fields are None
        when the stream is closed (idle player, between tracks).

        `sd_dtype` is one of "int16" / "int32" / "float32" matching
        sounddevice's reported sample type. We translate it to a
        nominal bit depth for the UI; the underlying PortAudio path
        may pack 24-bit samples into int32 containers, which the
        readout shows as "32-bit (int32)" to be honest about it.
        """
        with self._lock:
            stream = self._stream
            sample_rate = self._stream_sample_rate
            channels = self._stream_channels
            sd_dtype = self._stream_sd_dtype
            device_id = self._selected_device_id
            external = self._external_output_active
        device_name: Optional[str] = None
        if device_id:
            for entry in list_output_devices(stream is not None):
                if entry.get("id") == device_id:
                    device_name = entry.get("name")
                    break
        return {
            "stream_open": stream is not None,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sd_dtype": sd_dtype,
            "device_id": device_id or None,
            "device_name": device_name,
            "external_output_active": external,
        }

    def list_output_devices(self) -> list[dict]:
        """Enumerate output-capable audio devices for the picker UI.

        Thin wrapper around `output_devices.list_output_devices`:
        captures `stream_active` under the lock so the listing
        function knows whether it's safe to re-init PortAudio
        (which would tear down a live stream).

        See `app/audio/output_devices.py` for per-platform filter
        rationale and the picker payload shape.
        """
        with self._lock:
            stream_active = self._stream is not None
        return list_output_devices(stream_active)

    def set_output_device(self, device_id: str) -> None:
        """Remember the device-id selection and, if a stream is
        currently open, reopen it on the new device without
        dropping the current decoder/queue. Expected ~50ms of
        silence during the reopen — small blip, same as any
        device-switch in any audio app.
        """
        device_id = device_id or ""
        with self._pipeline_lock:
            with self._lock:
                prev_id = self._selected_device_id
                self._selected_device_id = device_id
                current_stream = self._stream
                sample_rate = self._stream_sample_rate
                channels = self._stream_channels
                sd_dtype = self._stream_sd_dtype
                was_playing = (
                    self._state == "playing"
                    and current_stream is not None
                )
            if (
                not was_playing
                or prev_id == device_id
                or sample_rate is None
                or channels is None
                or sd_dtype is None
            ):
                # Either nothing's playing (picked-up on next load),
                # the device didn't actually change, or we're missing
                # the info we'd need to reopen. Nothing more to do.
                return

            # Close the current stream + open fresh on the new
            # device. The decoder thread keeps filling the PCM
            # queue in the background, so the new stream picks up
            # where the old one left off after its first callback.
            #
            # `_replacing_stream` muzzles the finished_callback that
            # `stream.stop()` triggers — without it, the callback
            # would interpret the stop as a natural end-of-track and
            # transition state to "ended" mid-swap, briefly telling
            # the frontend playback finished.
            with self._lock:
                self._replacing_stream = True
            try:
                try:
                    current_stream.stop()
                except Exception:
                    pass
                _close_pa_stream(current_stream)
                with self._lock:
                    self._stream = None
                    # Reset the callback's mid-frame carry so the
                    # first post-reopen callback starts on a fresh
                    # chunk boundary — otherwise we could re-emit a
                    # partial chunk of already-played samples.
                    self._callback_carry = None
                    self._open_output_stream(sample_rate, channels, sd_dtype)
                try:
                    self._start_stream(self._stream)
                except Exception:
                    log.exception("set_output_device: stream start failed")
            finally:
                with self._lock:
                    self._replacing_stream = False

    def set_exclusive_mode(self, enabled: bool) -> None:
        """Flip exclusive-mode on the audio output stream. If a stream
        is open the reopen applies immediately (~50 ms of silence,
        same as a device switch); otherwise the flag kicks in on the
        next track load."""
        enabled = bool(enabled)
        with self._pipeline_lock:
            with self._lock:
                prev = self._exclusive_mode
                self._exclusive_mode = enabled
                current_stream = self._stream
                sample_rate = self._stream_sample_rate
                channels = self._stream_channels
                sd_dtype = self._stream_sd_dtype
                was_playing = (
                    self._state == "playing" and current_stream is not None
                )
            if (
                prev == enabled
                or not was_playing
                or sample_rate is None
                or channels is None
                or sd_dtype is None
            ):
                return

            # Same intentional-stop muzzle as set_output_device — see
            # the comment there.
            with self._lock:
                self._replacing_stream = True
            try:
                try:
                    current_stream.stop()
                except Exception:
                    pass
                _close_pa_stream(current_stream)
                with self._lock:
                    self._stream = None
                    self._callback_carry = None
                    self._open_output_stream(sample_rate, channels, sd_dtype)
                try:
                    self._start_stream(self._stream)
                except Exception:
                    log.exception("set_exclusive_mode: stream start failed")
            finally:
                with self._lock:
                    self._replacing_stream = False

    def snapshot(self) -> PlayerSnapshot:
        with self._lock:
            pos_ms = 0
            # While a DLNA renderer's clock drives a bounded track, the
            # local decode clock does not advance, so report the
            # renderer's own position instead. This is what lets the
            # frontend's position-gated preload (fire ~10s into the
            # track) run for the track after a renderer auto-advance, and
            # keeps the scrubber honest in DLNA mode.
            renderer_ms = None
            if _upnp_manager is not None:
                try:
                    renderer_ms = _upnp_manager.renderer_position_ms()
                except Exception:
                    renderer_ms = None
            if renderer_ms is not None:
                pos_ms = renderer_ms
            elif self._native is not None:
                pos_ms = self._native_pos_ms
                if (
                    self._native_playback_state == "Playing"
                    and self._state == "playing"
                ):
                    pos_ms += int((time.monotonic() - self._native_pos_at) * 1000)
            elif self._stream_sample_rate and self._samples_emitted > 0:
                pos_ms = int(
                    self._samples_emitted * 1000 / self._stream_sample_rate
                )
            # Clamp to duration so the UI's progress bar doesn't
            # overshoot when the callback reports samples beyond
            # the real track end (can happen for ~one frame).
            if self._current_duration_ms > 0:
                pos_ms = min(pos_ms, self._current_duration_ms)
            return PlayerSnapshot(
                state=self._state,
                track_id=self._current_track_id,
                position_ms=pos_ms,
                duration_ms=self._current_duration_ms,
                volume=self._volume,
                muted=self._muted,
                error=self._last_error,
                error_kind=(
                    "backoff"
                    if self._last_error and _tidal_backoff_active()
                    else None
                ),
                seq=self._seq,
                stream_info=self._current_stream_info,
                immersive=_dolby_playback_kind(
                    self._current_stream_info,
                    native=self._native is not None,
                    spatial_format=self._native_spatial_format,
                    decoders=self._native_decoders,
                ),
                force_volume=self._force_volume,
            )

    # --- internals --------------------------------------------------

    @staticmethod
    def _dispose_pipeline_async(thread, decoder) -> None:
        """Join a torn-down decoder thread and close its decoder, off the
        realtime / request path (see the call site in `_teardown`).

        Ordering is load-bearing: the thread must have exited the PyAV
        container before `close()` frees it, so we join first. The join
        is bounded — a permanently wedged read can't pin this disposer
        (and it's a daemon, so it never blocks process exit either). The
        decoder being disposed here is already detached from the player
        (`self._decoder`/`self._decoder_thread` were cleared before the
        spawn), so this can't race the next track's fresh pipeline."""
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if decoder is not None:
            try:
                decoder.close()
            except Exception:
                pass

    def _retire_pipeline(self, stop_flag, decoder, thread) -> None:
        """Stop a pipeline that is no longer the player's and release
        it in the background.

        For a track change, never for seek: the old thread must own a
        queue the audio callback no longer reads, because it may push
        one more chunk before it notices the stop flag. Seek reuses the
        live queue, which is why `_restart_decoder_at` joins inline.

        Joining inline here is what made a track change wait: a
        producer that is ahead of playback sits in `pcm_queue.put`
        with a 0.5 s timeout and only sees the stop flag when that
        expires, so the join cost up to half a second per pipeline on
        every click (`swap=~450ms` in the load perf line). Stopping is
        immediate: the flag is set, and cancel_source() aborts an
        in-flight segment fetch, which also releases the TIDAL reader
        right away (TIDAL drops the session with two tracks open).
        Only the join + close wait, on the disposer thread."""
        if stop_flag is not None:
            stop_flag.set()
        if decoder is not None:
            decoder.cancel_source()
        if thread is not None or decoder is not None:
            threading.Thread(
                target=self._dispose_pipeline_async,
                args=(thread, decoder),
                daemon=True,
                name="pcm-dispose",
            ).start()

    def _retire_preload(self, pre: Optional["_Preload"]) -> None:
        """`_retire_pipeline` for a preload (or the synthetic preload a
        gapless load builds). A cast preload is a metadata-only record
        with no thread or decoder, so there's nothing to release."""
        if pre is None or getattr(pre, "cast", False) is True:
            return
        self._retire_pipeline(pre.stop_flag, pre.decoder, pre.thread)

    def _teardown(
        self,
        *,
        keep_native: bool = False,
        park_native: bool = False,
        keep_dash: Optional[_DashAtmos] = None,
    ) -> Optional[NativeAtmosPlayer]:
        """Stop the pipeline fully. State transitions to idle BEFORE
        we close the stream, so the finished_callback short-circuits
        and doesn't emit a spurious 'ended' at user-initiated stops.
        Any preload is also dropped — a stale preload on a
        user-stopped pipeline would fire a bogus bridge when the
        next load() came in.

        `keep_native` hands the live helper back instead of closing
        it, paused. Spawning the PowerShell host and standing up
        MediaPlayer costs ~0.4-0.6 s (measured), which is pure
        overhead on an Atmos -> Atmos track change: the next track
        only needs `Source` pointed at a new file. The caller owns
        the returned helper and must either reuse it or close it.

        `park_native` is the mixed-queue counterpart: the outgoing
        helper is left idle in the warm slot instead of being shut
        down, so an Atmos -> PCM -> Atmos sequence spawns once, not
        twice. Only the process is kept — nothing audible.

        `keep_dash` likewise spares the one session feeding the track
        that is about to open; every other stream — including a
        look-ahead download for a track that is no longer next — is
        retired here, because its track is the one being torn down.
        """
        # Drop preload first so finished_callback's `pre` read sees
        # None when it fires during stream.close() below.
        self._drop_preload()
        self._pending_upnp_notify = False
        self._retire_dash_sessions(keep=keep_dash)
        if keep_native or park_native:
            # Both keep the idle helper: `keep_native` is about to adopt
            # the warmed look-ahead, `park_native` is handing the queue
            # to PCM. Only one that sat past its TTL — the same helper
            # `_take_warm_helper` would refuse — is closed.
            self._reap_idle_helper()
        else:
            # A stop leaves nothing for the warmed look-ahead to play.
            self._close_warm_helper()
        reuse: Optional[NativeAtmosPlayer] = None
        native, native_stop = self._clear_native_locked()
        if native is not None:
            keep = False
            if keep_native and getattr(native, "running", False):
                # Silence the outgoing track before the caller spends
                # time resolving the next one; assigning `Source`
                # later stops playback anyway, this just gets there
                # first and is cheap.
                try:
                    native.pause()
                except Exception as exc:
                    log.warning("native atmos pause on teardown failed: %s", exc)
                reuse = native
                keep = True
            elif park_native:
                keep = self._park_native_helper(native)
            if not keep:
                # Closing blocks briefly on the helper's quit handshake;
                # do it before touching the sounddevice stream so a native
                # track can't bleed into the next PCM track's open.
                try:
                    native.close()
                except Exception:
                    pass
        if native_stop is not None:
            native_stop.set()
        with self._lock:
            # If we're already idle, nothing to do.
            if self._state == "idle" and self._stream is None:
                return reuse
            self._state = "idle"
            self._seq += 1

        self._stop_flag.set()
        # Cancel the decoder's source BEFORE joining the thread.
        # The thread is very often mid-HTTP-fetch inside SegmentReader
        # when a track change fires, and stop_flag is only checked
        # between reads. Closing the source aborts the in-flight
        # request so read() raises on the thread and it exits in
        # single-digit ms instead of ~400ms.
        decoder = self._decoder
        if decoder is not None:
            try:
                decoder.cancel_source()
            except Exception:
                pass

        stream = self._stream
        self._stream = None
        if stream is not None:
            # abort() over stop(): stop() drains any buffered audio
            # to the device before returning, which on track-change
            # costs 400-900ms of nothing-is-happening. abort()
            # discards the pending buffer immediately and we start
            # the next track faster. Any in-flight tail audio just
            # gets cut off, which is what the user already expects
            # when clicking a new track.
            t_abort0 = time.monotonic()
            _close_pa_stream(stream)
            print(
                f"[perf] teardown stream.abort+close="
                f"{(time.monotonic() - t_abort0) * 1000.0:.0f}ms",
                file=sys.stderr,
                flush=True,
            )

        # The stream is aborted, so the audio callback can no longer fire —
        # safe now to cancel any in-progress crossfade and dispose the
        # incoming preload we'd claimed (a manual skip / stop landing inside
        # the fade window). No-op when no crossfade is active.
        self._abort_crossfade()

        # Dispose the old decoder thread + decoder OFF this path. The
        # join blocks until the thread unwinds its wedged Tidal segment
        # read — ~1s on a contended / slow network even though
        # cancel_source() above already aborted the in-flight request —
        # and _teardown runs on the request thread holding the GIL, so a
        # synchronous join here starves the realtime audio callback for
        # that whole second: an audible ~1s dropout at every track change
        # on a loaded machine (visible as the back-to-back
        # "[perf] teardown thread.join=~1300ms" + "callback jitter late
        # by ~770ms" pair in audio.log). Hand the join + ordered close
        # (close must follow the thread leaving the PyAV container) to a
        # daemon so the next track loads immediately; the old pipeline
        # finishes unwinding in the background.
        thread = self._decoder_thread
        self._decoder_thread = None
        self._decoder = None
        if thread is not None or decoder is not None:
            threading.Thread(
                target=self._dispose_pipeline_async,
                args=(thread, decoder),
                daemon=True,
                name="pcm-dispose",
            ).start()

        with self._lock:
            self._drain_queue_locked()
            self._callback_carry = None
            self._samples_emitted = 0
            self._current_track_id = None
            self._current_duration_ms = 0
            self._current_stream_info = None
            self._stream_sample_rate = None
            self._source_urls = None
            self._source_path = None
            self._current_track_meta = None
            self._paused = False
            self._seeking = False
        return reuse

    def _build_track_meta(self, track, duration_s) -> dict:
        """Extract UPnP/DLNA track metadata from a resolved Tidal track.

        Returns the metadata dict the UPnP layer consumes for
        SetAVTransportURI. `cover_id` is carried separately so the
        UPnP layer can build a LAN-local cover URL — the renderer can't
        reach Tidal's image CDN directly, which is why the cover never
        showed before.
        """
        artist_name = (
            getattr(track.artist, "name", None)
            if hasattr(track, "artist")
            else None
        )
        if not artist_name and hasattr(track, "artists") and track.artists:
            artist_name = track.artists[0].name
        album_name = (
            track.album.name
            if hasattr(track, "album") and track.album
            else None
        )
        cover_id = (
            track.album.cover
            if hasattr(track, "album") and track.album
            else None
        )
        return {
            "title": getattr(track, "name", "") or "",
            "artist": artist_name or "",
            "album": album_name or "",
            "duration_s": duration_s or 0,
            "cover_id": cover_id or "",
            "cover_url": (
                f"https://resources.tidal.com/images/{cover_id}/640x640.jpg"
                if cover_id
                else ""
            ),
        }

    def _resolve_source(
        self,
        track_id: str,
        quality: Optional[str],
        *,
        set_current_meta: bool = True,
    ) -> tuple[Union[str, list[str]], Optional[float], StreamInfo, dict[int, bytes], Optional[dict]]:
        """Resolve a track id to a source spec.

        Returns either a local filesystem path (str) or a DASH URL
        list (list[str] — index 0 is the init segment). The caller
        constructs a `Decoder` from it via `_build_source()` and
        stores the spec on the player so seek can rebuild.

        The fourth return value is a map of pre-fetched segment
        bytes (idx -> bytes). Empty for local files and for stream
        sources that have not had byte-level prefetch run against
        them. When populated, SegmentReader seeds its buffer with
        those bytes so decoder_init does not touch the network.

        `set_current_meta` is passed False by speculative resolution
        (cold-start / hover prefetch): those warm the cache but must
        NOT overwrite `_current_track_meta`. The renderer's first
        SetAVTransportURI reads that dict, so a prefetch of a track
        the user never played would otherwise show up as the wrong
        "current" track at the start of a session.
        """
        if self._local_lookup is not None:
            local_path = self._local_lookup(track_id)
            if local_path and os.path.exists(local_path):
                info, duration_s = _probe_local_stream_info(local_path)
                return (
                    local_path,
                    duration_s,
                    info or StreamInfo(source="local"),
                    {},
                    None,
                )

        session = self._session_getter()
        if quality and self._quality_clamp is not None:
            try:
                clamped = self._quality_clamp(quality)
                if clamped and clamped != quality:
                    log.info(
                        "clamped streaming quality %r -> %r", quality, clamped
                    )
                    quality = clamped
            except Exception:
                pass

        # Finished file from a previous play: open it directly. This
        # skips the TIDAL round-trips and the download, and a real
        # file seeks precisely. Skipped while casting (the renderer
        # plays TIDAL URLs, not local files) and for Atmos, which has
        # the native pipeline's own cache.
        t0 = time.monotonic()
        if (
            quality
            and quality.lower() != "atmos"
            and not (
                _upnp_manager is not None and _upnp_manager.is_active()
            )
        ):
            disk_hit = self._pcm_stream_cache_lookup(track_id, quality)
            if disk_hit is not None and not _cached_tier_is_usable(
                quality, disk_hit[2]
            ):
                log.info(
                    "ignoring %s disk-cache entry for %s: it holds a "
                    "lossy stream under a lossless key",
                    quality,
                    track_id,
                )
                disk_hit = None
            if disk_hit is not None:
                path, duration_s, info, meta = disk_hit
                if meta is not None and set_current_meta:
                    self._current_track_meta = meta
                print(
                    f"[perf] resolve track={track_id} quality={quality} "
                    f"cache=DISK elapsed="
                    f"{(time.monotonic() - t0) * 1000.0:.0f}ms "
                    f"prefetched_segments=0",
                    file=sys.stderr,
                    flush=True,
                )
                return path, duration_s, info, {}, meta

        # Cache hit path: skip the two parallel Tidal round-trips
        # (track metadata + playbackinfo) plus the local manifest
        # parse when a recent prefetch or play already resolved this
        # (track_id, quality) pair. 3-minute TTL stays well inside
        # Tidal's signed-URL expiry window.
        cache_key = (track_id, quality)
        cached_urls_info = self._manifest_cache.lookup(cache_key)
        if cached_urls_info is not None and not _cached_tier_is_usable(
            quality, cached_urls_info[2]
        ):
            cached_urls_info = None
        if cached_urls_info is not None:
            return self._resolved_from_cache(
                cached_urls_info, track_id, quality, t0, set_current_meta, "HIT"
            )
        # A hover prefetch is often still resolving this exact track
        # when the click lands. Joining that resolve is what makes a
        # quick hover-then-click fast: the click waits out the tail of
        # work already in flight instead of repeating the same Tidal
        # round-trips from scratch and paying them twice over.
        ticket, leader = self._claim_resolve(cache_key)
        if not leader:
            ticket.event.wait(_RESOLVE_JOIN_S)
            cached_urls_info = self._manifest_cache.lookup(cache_key)
            if cached_urls_info is not None and not _cached_tier_is_usable(
                quality, cached_urls_info[2]
            ):
                cached_urls_info = None
            if cached_urls_info is not None:
                return self._resolved_from_cache(
                    cached_urls_info, track_id, quality, t0, set_current_meta, "JOIN"
                )
            if ticket.result is not None:
                spec, duration, info, bytes_map, track_meta = ticket.result
                if track_meta is not None and set_current_meta:
                    self._current_track_meta = track_meta
                return (spec, duration, info, dict(bytes_map), track_meta)
            # A failed or evicted leader leaves no answer behind. Claim
            # the key again so this caller retries instead of inheriting
            # a transient failure from the hover or preload it joined.
            ticket, leader = self._claim_resolve(cache_key)

        try:
            result = self._resolve_miss(
                session, track_id, quality, cache_key, t0,
                set_current_meta=set_current_meta,
            )
        except BaseException as exc:
            self._release_resolve(
                cache_key, ticket if leader else None, error=exc
            )
            raise
        self._release_resolve(cache_key, ticket if leader else None, result=result)
        return result

    def _resolved_from_cache(
        self,
        entry: tuple,
        track_id: str,
        quality: Optional[str],
        t0: float,
        set_current_meta: bool,
        label: str,
    ) -> tuple[Union[str, list[str]], Optional[float], StreamInfo, dict[int, bytes], Optional[dict]]:
        """Return a cached resolve, refreshing the UPnP metadata.

        The miss path sets `_current_track_meta` as part of resolving;
        a hit used to leave it pointing at the previous track, so the
        renderer was told the wrong artist/song at session start (and
        on every replay, since a prefetch makes the first play a hit).
        Guarded by `set_current_meta` so speculative prefetch does not
        clobber it.
        """
        urls, duration, info, bytes_map, track_meta = entry
        if track_meta is not None and set_current_meta:
            self._current_track_meta = track_meta
        print(
            f"[perf] resolve track={track_id} quality={quality} "
            f"cache={label} elapsed={(time.monotonic() - t0) * 1000.0:.0f}ms "
            f"prefetched_segments={len(bytes_map)}",
            file=sys.stderr,
            flush=True,
        )
        return (list(urls), duration, info, bytes_map, track_meta)

    def _pcm_stream_cache_lookup(
        self, track_id: str, quality: Optional[str]
    ) -> Optional[tuple[str, Optional[float], StreamInfo, Optional[dict]]]:
        """A finished file for this (track, quality), or None.

        The caller has already excluded Atmos and cast sessions; this
        is only about reading the sidecar back into a StreamInfo.
        """
        cache = getattr(self, "_pcm_stream_cache", None)
        if cache is None:
            return None
        try:
            hit = cache.lookup(track_id, quality)
        except OSError:
            return None
        if hit is None:
            return None
        path, duration_s, payload, meta = hit
        try:
            info = StreamInfo(**payload)
        except TypeError:
            # Written by a build whose StreamInfo had different
            # fields; treat it as a miss and download again.
            return None
        return path, duration_s, info, meta

    def _pcm_stream_cache_sink(
        self,
        track_id: str,
        quality: Optional[str],
        spec: Union[str, list[str]],
        duration_s: Optional[float],
        info: StreamInfo,
        meta: Optional[dict],
    ) -> Optional[object]:
        """A disk-cache tee for this play's segments, or None.

        Caching applies only to a real TIDAL stream at a concrete
        quality. A local file is already on disk, Atmos has its own
        cache, a cast session hands the renderer TIDAL URLs (so the
        local decoder is not what fetches the track), and a
        quality-less request's effective tier is unknown until the
        manifest arrives — keying on it could serve the wrong tier.
        """
        if not isinstance(spec, list) or not quality:
            return None
        if quality.lower() == "atmos":
            return None
        if _upnp_manager is not None and _upnp_manager.is_active():
            return None
        cache = getattr(self, "_pcm_stream_cache", None)
        if cache is None:
            return None
        try:
            return cache.sink(
                track_id,
                quality,
                len(spec),
                duration_s,
                asdict(info),
                meta,
            )
        except OSError:
            return None

    def _claim_resolve(self, cache_key: tuple) -> tuple[Optional["_ResolveTicket"], bool]:
        """Register interest in a resolve, or find one to wait on.

        Returns (ticket, is_leader). A player built without `__init__`
        (unit tests) has no registry: every caller is its own leader.
        """
        lock = getattr(self, "_resolving_lock", None)
        if lock is None:
            return None, True
        with lock:
            ticket = self._resolving.get(cache_key)
            if ticket is not None:
                return ticket, False
            ticket = _ResolveTicket()
            self._resolving[cache_key] = ticket
            return ticket, True

    def _release_resolve(
        self,
        cache_key: tuple,
        ticket: Optional["_ResolveTicket"],
        *,
        result: Optional[tuple] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Publish a resolve's outcome and wake the threads waiting on it."""
        if ticket is None:
            return
        ticket.result = result
        ticket.error = error
        with self._resolving_lock:
            if self._resolving.get(cache_key) is ticket:
                self._resolving.pop(cache_key, None)
        ticket.event.set()

    def _resolve_miss(
        self,
        session: tidalapi.Session,
        track_id: str,
        quality: Optional[str],
        cache_key: tuple,
        t0: float,
        *,
        set_current_meta: bool = True,
        _refreshed: bool = False,
    ) -> tuple[Union[str, list[str]], Optional[float], StreamInfo, dict[int, bytes], Optional[dict]]:
        try:
            return self._resolve_uncached(
                session, track_id, quality, cache_key, t0,
                set_current_meta=set_current_meta,
            )
        except atmos_mod.AtmosUnavailable as exc:
            # The user asked for Atmos but this release has no E-AC-3 JOC
            # asset (or the manifest was unusable). Fall back to the best
            # lossless tier instead of failing the play outright — the
            # now-playing pill reports the quality that's actually
            # audible, so the downgrade is visible, not silent-but-wrong.
            log.info(
                "Dolby Atmos unavailable for track %s (%s); falling back "
                "to hi_res_lossless",
                track_id,
                exc,
            )
            fallback_quality = "hi_res_lossless"
            if self._quality_clamp is not None:
                try:
                    clamped = self._quality_clamp(fallback_quality)
                    if clamped:
                        fallback_quality = clamped
                except Exception:
                    pass
            fallback_key = (track_id, fallback_quality)
            cached = self._manifest_cache.lookup(fallback_key)
            if cached is not None and not _cached_tier_is_usable(
                fallback_quality, cached[2]
            ):
                cached = None
            if cached is not None:
                urls, duration, info, bytes_map, track_meta = cached
                if track_meta is not None and set_current_meta:
                    self._current_track_meta = track_meta
                return (list(urls), duration, info, bytes_map, track_meta)
            try:
                return self._resolve_uncached(
                    session,
                    track_id,
                    fallback_quality,
                    fallback_key,
                    t0,
                    set_current_meta=set_current_meta,
                )
            except Exception as fallback_exc:
                # An error raised *inside* this handler would otherwise
                # skip the auth-retry below — sibling `except` clauses
                # never see it. Expired-token 401s on the lossless
                # fallback need the same refresh-and-retry treatment as
                # a direct resolve, which is why the original quality
                # (not the fallback) is retried: the refreshed session
                # may well be entitled to Atmos.
                return self._retry_after_auth_error(
                    fallback_exc,
                    track_id,
                    quality,
                    cache_key,
                    t0,
                    set_current_meta,
                    _refreshed,
                )
        except Exception as exc:
            return self._retry_after_auth_error(
                exc,
                track_id,
                quality,
                cache_key,
                t0,
                set_current_meta,
                _refreshed,
            )

    def _retry_after_auth_error(
        self,
        exc: BaseException,
        track_id: str,
        quality: Optional[str],
        cache_key: tuple,
        t0: float,
        set_current_meta: bool,
        refreshed: bool,
    ) -> tuple[Union[str, list[str]], Optional[float], StreamInfo, dict[int, bytes], Optional[dict]]:
        """Refresh an expired Tidal token once and redo the resolve.

        The playback resolve path must recover from an expired token
        exactly like the download and metadata paths already do.
        Without this, a stale access token surfaces as a silent "press
        play, nothing happens": the 401 propagates, the player flips to
        error, and the user is neither refreshed nor bounced to Login.

        `getattr` (not `self._force_refresh`) because some unit tests
        build PCMPlayer via `__new__` and never run `__init__`; a
        missing injector just means "no refresh".
        """
        force_refresh = getattr(self, "_force_refresh", None)
        if (
            not refreshed
            and force_refresh is not None
            and _looks_like_auth_error(exc)
        ):
            if force_refresh():
                log.info(
                    "playback resolve hit a Tidal auth error; "
                    "token refreshed, retrying once"
                )
                return self._resolve_miss(
                    self._session_getter(),
                    track_id,
                    quality,
                    cache_key,
                    t0,
                    set_current_meta=set_current_meta,
                    _refreshed=True,
                )
            raise RuntimeError(
                "Tidal session expired. Please log out and log back in."
            ) from exc
        raise exc

    def _resolve_uncached(
        self,
        session: tidalapi.Session,
        track_id: str,
        quality: Optional[str],
        cache_key: tuple,
        t0: float,
        *,
        set_current_meta: bool = True,
    ) -> tuple[
        Union[str, list[str]], Optional[float], StreamInfo, dict[int, bytes]
    ]:
        """Cache-miss network resolution: fetch track + playback-info,
        parse the manifest, populate the cache. Split out of
        _resolve_source so the caller can retry it once after a
        force-refresh when Tidal returns an expired-token 401.

        `set_current_meta` mirrors _resolve_source: speculative
        resolution (prefetch) passes False so it doesn't overwrite the
        metadata of the track that is actually playing."""
        if (quality or "").lower() == "atmos":
            # Atmos is not a tidalapi.Quality member; it has its own
            # OpenAPI manifest path. Handle it before the v1 quality
            # override so we never set session.config.quality to a
            # nonexistent tier.
            return self._resolve_atmos(
                session, track_id, cache_key, t0,
                set_current_meta=set_current_meta,
            )
        override = None
        if quality:
            try:
                override = tidalapi.Quality[quality]
            except KeyError:
                log.warning("unknown quality %r, using session default", quality)
        original = session.config.quality
        if override is not None:
            session.config.quality = override
        try:
            # The metadata fetch (`session.track`) and the playback-info
            # fetch (`track.get_stream`) are two independent Tidal
            # round-trips: get_stream only uses the track id, not any
            # parsed metadata. Run them in parallel so total wall-clock
            # is max() instead of sum() — saves ~50-150 ms per cache
            # miss on a click. The third historical "phase",
            # `stream.get_stream_manifest()`, is a local base64 +
            # parse, not a network call.
            tid = int(track_id)
            t_start = time.monotonic()
            # Not a `with` block: ThreadPoolExecutor.__exit__ does
            # shutdown(wait=True), which would re-block on a hung
            # future even after result(timeout=) gave up — defeating
            # the whole point of bounding the wait. shutdown(wait=
            # False, cancel_futures=True) lets us return promptly; the
            # transport timeout (app.http.DEFAULT_TIMEOUT) ensures the
            # orphaned worker actually unwinds shortly after instead
            # of lingering.
            # Shared with the downloader: TIDAL's legacy /urlpostpaywall
            # path (and the LOSSLESS tier itself, on some clients)
            # silently answers AAC 320 even for a lossless request, so
            # resolution has to inspect the delivered tier and escalate
            # to HI_RES when that's what it takes to get FLAC.
            from app.tidal_client import request_stream_manifest

            ex = ThreadPoolExecutor(max_workers=2)
            try:
                track_future = ex.submit(session.track, tid)
                stream_holder = tidalapi.Track(session)
                stream_holder.id = tid
                stream_future = ex.submit(
                    request_stream_manifest, session, stream_holder
                )
                deadline = time.monotonic() + _RESOLVE_TIMEOUT_S
                try:
                    stream, manifest, delivered = stream_future.result(
                        timeout=_RESOLVE_TIMEOUT_S
                    )
                except FutureTimeoutError:
                    # Surfaces to _load_locked's except: it flips the
                    # player to state=error and emits, so the frontend
                    # clears the spinner and the user can pick another
                    # track — instead of a permanent "loading" that
                    # only an app restart escaped.
                    raise RuntimeError(
                        "Tidal did not respond while resolving the "
                        "track (timed out)"
                    )
                try:
                    track = track_future.result(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except Exception as exc:  # noqa: BLE001 - metadata is optional
                    # The playback service is authoritative: ids that
                    # exist only in the v2 catalog (or that the v1
                    # catalog dropped) 404 on `tracks/{id}` while still
                    # streaming fine. Failing the whole play here is
                    # what made a previously cached track unplayable
                    # once its cache was deleted.
                    log.info(
                        "track metadata unavailable for %s (%s); "
                        "playing without it",
                        track_id,
                        exc,
                    )
                    track = None
            finally:
                ex.shutdown(wait=False, cancel_futures=True)
            t_end = time.monotonic()
            if getattr(manifest, "is_encrypted", False):
                raise RuntimeError("encrypted stream — can't decode")
            urls = list(getattr(manifest, "urls", []) or [])
            if not urls:
                raise RuntimeError("manifest has no segment urls")
            duration = getattr(track, "duration", None) if track is not None else None
            # Resolve track metadata for UPnP/DLNA track-change
            # notification. Also stored on the cache entry (below) so a
            # later cache HIT carries the correct metadata too. Only for
            # real loads — speculative prefetch passes set_current_meta
            # False so it can't clobber the playing track's metadata.
            track_meta = (
                self._build_track_meta(track, duration)
                if track is not None
                else None
            )
            if track_meta is not None and set_current_meta:
                self._current_track_meta = track_meta
            info = StreamInfo(
                source="stream",
                codec=_normalize_codec(
                    getattr(manifest, "codecs", None)
                    or getattr(manifest, "get_codecs", lambda: None)()
                ),
                bit_depth=_safe_int(getattr(stream, "bit_depth", None)),
                sample_rate_hz=_safe_int(getattr(stream, "sample_rate", None)),
                audio_quality=getattr(stream, "audio_quality", None),
                audio_mode=getattr(stream, "audio_mode", None),
                # tidalapi exposes the EBU R128 ReplayGain values + actual
                # peaks straight off the Stream object. Storing them on
                # StreamInfo means the audio engine can read them through
                # the same channel everything else flows through (cache
                # entry, preload, current track) without a separate
                # plumbing path.
                track_replay_gain_db=_safe_float(
                    getattr(stream, "track_replay_gain", None)
                ),
                track_peak=_safe_float(
                    getattr(stream, "track_peak_amplitude", None)
                ),
                album_replay_gain_db=_safe_float(
                    getattr(stream, "album_replay_gain", None)
                ),
                album_peak=_safe_float(
                    getattr(stream, "album_peak_amplitude", None)
                ),
            )
            duration_s = float(duration) if duration else None
            self._manifest_cache.store(
                cache_key, list(urls), duration_s, info, track_meta
            )
            # After store, any previously-warmed bytes are preserved
            # — pick them back up so a MISS on manifest that still
            # has bytes cached returns the bytes too.
            cached = self._manifest_cache.lookup(cache_key)
            bytes_map = cached[3] if cached is not None else {}
            print(
                f"[perf] resolve track={track_id} quality={quality} cache=MISS "
                f"total={(t_end - t0) * 1000.0:.0f}ms "
                f"network_parallel={(t_end - t_start) * 1000.0:.0f}ms "
                f"delivered={delivered} "
                f"segments={len(urls)}",
                file=sys.stderr,
                flush=True,
            )
            return (list(urls), duration_s, info, bytes_map, track_meta)
        finally:
            if override is not None:
                session.config.quality = original

    def _resolve_atmos(
        self,
        session: tidalapi.Session,
        track_id: str,
        cache_key: tuple,
        t0: float,
        *,
        set_current_meta: bool = True,
    ) -> tuple[
        list[str], Optional[float], StreamInfo, dict[int, bytes], Optional[dict]
    ]:
        """Resolve the Dolby Atmos (E-AC-3 JOC) asset for `track_id`.

        Runs the whole catalog dance — metadata, twin lookup, OpenAPI
        manifest, DASH expansion — on one worker thread so a hung Tidal
        request is bounded by `_RESOLVE_TIMEOUT_S` rather than pinning
        the pipeline lock. Raises `AtmosUnavailable` when this release
        has no Atmos asset; `_resolve_source` turns that into a lossless
        fallback.
        """
        tid = int(track_id)
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            future = ex.submit(atmos_mod.resolve_atmos_playback, session, tid)
            try:
                track, stream = future.result(timeout=_RESOLVE_TIMEOUT_S)
            except FutureTimeoutError:
                raise RuntimeError(
                    "Tidal did not respond while resolving the Dolby "
                    "Atmos stream (timed out)"
                )
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        # The resolver reports None when Tidal's v1 catalog doesn't know
        # the id even though the manifest service serves the stream;
        # duration and renderer metadata are then simply unknown.
        duration = getattr(track, "duration", None) if track is not None else None
        track_meta = (
            self._build_track_meta(track, duration) if track is not None else None
        )
        if track_meta is not None and set_current_meta:
            self._current_track_meta = track_meta

        info = StreamInfo(
            source="stream",
            codec=_normalize_codec(stream.codec) or "eac3",
            bit_depth=stream.bit_depth,
            sample_rate_hz=stream.sample_rate,
            audio_quality="DOLBY_ATMOS",
            audio_mode="DOLBY_ATMOS",
            track_replay_gain_db=stream.track_replay_gain,
            track_peak=stream.track_peak,
            album_replay_gain_db=stream.album_replay_gain,
            album_peak=stream.album_peak,
        )
        duration_s = float(duration) if duration else None
        self._manifest_cache.store(
            cache_key, list(stream.urls), duration_s, info, track_meta
        )
        cached = self._manifest_cache.lookup(cache_key)
        bytes_map = cached[3] if cached is not None else {}
        print(
            f"[perf] resolve track={track_id} quality=atmos cache=MISS "
            f"total={(time.monotonic() - t0) * 1000.0:.0f}ms "
            f"segments={len(stream.urls)} channels={stream.channels} "
            f"rate={stream.sample_rate}",
            file=sys.stderr,
            flush=True,
        )
        return (list(stream.urls), duration_s, info, bytes_map, track_meta)

    def cache_stats(self) -> dict:
        """Snapshot of the manifest cache for the /api/player/cache-stats
        endpoint. Read while testing the prefetch path to confirm
        hovers / album-mount prefetches are landing. `stream` is the
        on-disk cache of completed non-Atmos tracks."""
        try:
            stream = self._pcm_stream_cache.stats()
        except Exception as exc:  # cache stats must never break the endpoint
            stream = {"error": str(exc)}
        return {**self._manifest_cache.stats(), "stream": stream}

    def prefetch(
        self, track_id: str, quality: Optional[str] = None, *, warm_bytes: bool = True
    ) -> bool:
        """Populate the manifest cache for a track without starting
        playback. Safe to call speculatively from hover / album-mount
        warmers. Returns True if the cache now has an entry for this
        (track_id, quality), False on any failure — callers are
        fire-and-forget so errors are swallowed.

        When warm_bytes=True (default), follows up by downloading the
        init segment plus the first media segment in parallel and
        caches those bytes too, so the next click's decoder_init
        can skip the network entirely.

        Jitters 50-200ms before the first Tidal request. Album-mount
        prefetch fans out via a ThreadPoolExecutor: without jitter, all
        N workers hit `_resolve_source` within the same few ms, and
        each one then issues two parallel Tidal API calls (track
        metadata + playbackinfo). That stack of N*2 requests inside
        one second is exactly the burst pattern Tidal's anti-abuse
        layer flags as a 429 (with longer 403/`abuse_detected`
        escalations from there). The jitter is per-worker, so the
        first request from each worker spreads across a 150 ms window
        before the parallel pair even starts. Foreground play goes
        through the same `_resolve_source` path so it benefits from
        the parallelization too, but doesn't go through this jitter
        wrapper since there's only one foreground request at a time.

        No-ops when prefetch is disabled by the caller — the endpoint
        layer bails out on offline_mode before ever reaching this
        method, so we only get here when prefetch is wanted."""
        # Lazy import: tidal_client doesn't import player, but
        # importing it from module-load time would still couple the
        # two trees together for no reason. The function is a tiny
        # `time.sleep(uniform(...))`, so the import cost is paid once
        # per process at first prefetch and never again.
        from app.tidal_client import tidal_jitter_sleep

        tidal_jitter_sleep()
        try:
            source_spec, _dur, _info, _bytes, _ = self._resolve_source(
                track_id, quality, set_current_meta=False
            )
        except Exception as exc:
            log.debug("prefetch manifest %s@%s failed: %s", track_id, quality, exc)
            return False
        if warm_bytes and isinstance(source_spec, list):
            self._warm_bytes((track_id, quality), source_spec)
        return True

    def _warm_bytes(
        self, key: tuple[str, Optional[str]], urls: list[str]
    ) -> None:
        """Best-effort download of the init + first media segments
        in parallel. Skips segments already in the cache so repeat
        hover events don't re-download. Fire-and-forget: any failure
        leaves the cache at URL-only and the eventual play path
        falls through to the normal SegmentReader fetch.

        A single-URL source is one whole progressive file (the AAC
        tiers), so warming it used to download every byte of the
        track — albums warm on mount, which meant a page open pulled
        tens of MB and starved the audio callback. Only a bounded
        header prefix is fetched for those; SegmentReader continues
        the rest from where the seed ends when the track is played.

        The fetches are also registered under `key` so a click that
        lands mid-download takes these bytes (`_adopt_warm_bytes`)
        instead of pulling the same segment a second time on the
        critical path. Each segment is published to the cache as soon
        as it lands, so an early click gets whatever is already in."""
        cached = self._manifest_cache.lookup(key)
        have = cached[3] if cached is not None else {}
        targets = [i for i in (0, 1) if i < len(urls) and i not in have]
        if not targets:
            return
        fetched: dict[int, bytes] = {}
        single = len(urls) == 1

        def _fetch(i: int) -> Optional[tuple[int, bytes]]:
            try:
                # The playback reader's pool, so warming a track also
                # leaves a live CDN connection for the click to reuse.
                headers = (
                    {"Range": f"bytes=0-{_WARM_HEAD_BYTES - 1}"}
                    if single
                    else None
                )
                r = STREAM_SESSION.get(
                    urls[i],
                    timeout=30,
                    headers=headers,
                    stream=single,
                )
                try:
                    r.raise_for_status()
                    if not single:
                        return i, r.content
                    # A CDN that ignores Range answers 200 with the
                    # whole body; keep only the prefix we asked for.
                    chunks: list[bytes] = []
                    got = 0
                    for chunk in r.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        chunks.append(chunk)
                        got += len(chunk)
                        if got >= _WARM_HEAD_BYTES:
                            break
                    return i, b"".join(chunks)[:_WARM_HEAD_BYTES]
                finally:
                    try:
                        r.close()
                    except Exception:
                        pass
            except Exception as exc:
                log.debug("prefetch bytes seg=%d failed: %s", i, exc)
                return None

        lock = getattr(self, "_warm_lock", None)
        with ThreadPoolExecutor(
            max_workers=min(2, len(targets)),
            thread_name_prefix="warm-bytes",
        ) as pool:
            futures = {i: pool.submit(_fetch, i) for i in targets}
            if lock is not None:
                with lock:
                    self._warm_inflight.setdefault(key, {}).update(futures)
            try:
                for i, future in futures.items():
                    result = future.result()
                    if result is None:
                        continue
                    idx, content = result
                    fetched[idx] = content
                    self._manifest_cache.update_bytes(key, {idx: content})
            finally:
                if lock is not None:
                    with lock:
                        pending = self._warm_inflight.get(key)
                        if pending is not None:
                            for i, future in futures.items():
                                # Identity, not index: a later hover may
                                # have registered its own fetch for the
                                # same segment by now.
                                if pending.get(i) is future:
                                    pending.pop(i, None)
                            if not pending:
                                self._warm_inflight.pop(key, None)
        if fetched:
            total = sum(len(v) for v in fetched.values())
            print(
                f"[perf] prefetch bytes track={key[0]} quality={key[1]} "
                f"segments={sorted(fetched.keys())} "
                f"total={total}B",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _start_stream(stream) -> None:
        """Start an output stream from whichever thread is asking.
        Every stream start goes through here: WASAPI needs COM on the
        starting thread, and these run on HTTP workers, the bridge and
        recovery threads (see app/audio/win_com.py)."""
        ensure_com_initialized()
        stream.start()

    def _adopt_warm_bytes(
        self,
        key: tuple[str, Optional[str]],
        prefetched: Optional[dict[int, bytes]],
    ) -> Optional[dict[int, bytes]]:
        """Take the bytes a hover prefetch is downloading right now.

        A click that arrives while the prefetch's init/first-segment
        fetches are in flight used to start a second download of the
        same segment for the DASH prime, and the two competed for the
        same connection. Waiting out the prefetch's remaining bytes is
        both faster and smaller: the prime then starts with the
        segments already in hand.
        """
        lock = getattr(self, "_warm_lock", None)
        if lock is None:
            return prefetched
        with lock:
            pending = dict(self._warm_inflight.get(key) or {})
        if not pending:
            return prefetched
        merged = dict(prefetched or {})
        deadline = time.monotonic() + _WARM_BYTES_JOIN_S
        for index, future in pending.items():
            if index in merged:
                continue
            try:
                result = future.result(
                    timeout=max(0.0, deadline - time.monotonic())
                )
            except Exception:
                continue
            if result:
                merged[result[0]] = result[1]
        return merged

    @staticmethod
    def _pa_open_stream(stream_kwargs: dict) -> "sd.OutputStream":
        """Open an OutputStream through pa_guard so its handle is
        tracked for teardown and reinit decisions."""
        return pa_guard.open_stream(**stream_kwargs)

    def _open_output_stream(
        self,
        sample_rate: int,
        channels: int,
        dtype: str,
        device_override: Optional[int] = None,
    ) -> None:
        # If the user picked a specific output device, resolve its
        # saved name to a live PortAudio index; otherwise device=None
        # routes to the system default. When the chosen device is no
        # longer present as an output (unplugged, or a legacy numeric
        # id from before device identity was name-based), fall back to
        # the default instead of hard-failing the load. Issue #245 was
        # a drifted index landing on an input-only device, where
        # sd.OutputStream raised "Invalid number of channels".
        #
        # `device_override` pins one enumeration entry for recovery:
        # the same physical endpoint is exposed through several host
        # APIs and they fail independently, so a retry may need to
        # open the MME copy instead of the WASAPI one the picker shows.
        if device_override is not None:
            device: Optional[int] = device_override
        else:
            device, available = resolve_output_device(self._selected_device_id)
            if self._selected_device_id and not available:
                print(
                    f"[audio] saved output device "
                    f"{self._selected_device_id!r} is not available; "
                    "falling back to system default",
                    flush=True,
                )

        # Decide the rate we'll actually feed sounddevice.
        # ─ Exclusive Mode: push the source rate straight at the device.
        #   The OS or driver is asked to reconfigure to that rate, and
        #   either succeeds (bit-perfect) or fails (we fall back below).
        # ─ Shared mode: ask the device for its mixer rate and resample
        #   to that rate inside the decoder. The OS then receives audio
        #   that already matches its mixer's rate, so its own resampler
        #   is a no-op and the only resampler in the chain is ours,
        #   where we control the headroom. When the source rate already
        #   matches the device's mixer rate (e.g. 44.1 source on a 44.1
        #   device), set_target_rate is a no-op and we stay bit-perfect
        #   even in shared mode.
        decoder = self._decoder
        device_max: Optional[int] = None
        if decoder is not None:
            if self._exclusive_mode:
                target_rate = decoder.sample_rate
            else:
                target_rate = self._query_device_mixer_rate(device, decoder.sample_rate)
            decoder.set_target_rate(
                target_rate,
                float_output=self._float_output_for(self._exclusive_mode),
            )
            sample_rate = decoder.output_sample_rate
            dtype = decoder.sounddevice_dtype
            channels = decoder.channels
            # Multichannel (Atmos) output only when the device actually
            # advertises the channels. PortAudio reports the device's
            # max_output_channels, which is what the OS mixer is
            # configured for; asking for more raises at open time, so
            # check first and downmix instead of failing the load. On
            # Windows shared mode the endpoint's bed is also what
            # Spatial Sound virtualises, so a 5.1 source is padded up
            # to a 7.1 endpoint rather than left short.
            if channels > 2:
                device_max = self._device_max_output_channels(device)
                windows = sys.platform.startswith("win")
                target_channels = _resolve_surround_channels(
                    decoder.source_channels,
                    device_max,
                    windows=windows,
                    exclusive=self._exclusive_mode,
                )
                if (
                    windows
                    and not self._exclusive_mode
                    and target_channels < channels
                    and device_override is None
                ):
                    # A stereo WASAPI mix format does not mean the
                    # endpoint can't take the bed: Windows' spatial APO
                    # holds a 5.1/7.1 bed internally while the mix
                    # format it advertises stays stereo. The legacy
                    # MME/DirectSound copies of the same device accept
                    # the layout and hand it to that APO, which is why
                    # VLC (waveout) gets surround on this endpoint and
                    # a WASAPI-only player does not. Re-open the same
                    # logical device through one of those host APIs
                    # before falling back to a downmix.
                    alt_device, alt_hostapi = resolve_surround_output_device(
                        self._selected_device_id,
                        channels,
                        sample_rate,
                        dtype,
                    )
                    if alt_device is not None and alt_device != device:
                        log.info(
                            "Windows spatial: routing %d-channel bed to "
                            "device %d (hostapi=%s) instead of downmixing "
                            "on the %d-channel WASAPI copy",
                            channels,
                            alt_device,
                            alt_hostapi,
                            device_max,
                        )
                        print(
                            f"[audio] spatial: device {device} "
                            f"({device_max}ch) cannot carry "
                            f"{channels}ch; using device {alt_device} "
                            f"hostapi={alt_hostapi!r}",
                            flush=True,
                        )
                        device = alt_device
                        device_max = self._device_max_output_channels(device)
                        target_channels = _resolve_surround_channels(
                            decoder.source_channels,
                            device_max,
                            windows=windows,
                            exclusive=self._exclusive_mode,
                        )
                if target_channels != channels:
                    if target_channels < channels:
                        log.info(
                            "device supports %d output channels; downmixing "
                            "%d-channel source to %d",
                            device_max,
                            channels,
                            target_channels,
                        )
                        if windows and device_max is not None and device_max <= 2:
                            log.info(
                                "Windows Spatial Sound is not seeing a "
                                "multichannel endpoint; enable Spatial sound "
                                "on this output (or pick the spatial device) "
                                "and reload to virtualise the surround bed"
                            )
                    else:
                        log.info(
                            "Windows Spatial Sound: padding %d-channel source "
                            "to the endpoint's %d-channel bed",
                            channels,
                            target_channels,
                        )
                    decoder.set_target_channels(target_channels)
                    channels = decoder.channels

        # Exclusive Mode — push PCM straight at the device at its
        # native rate / bit depth. On macOS the CoreAudio flags ask
        # the driver to reconfigure the device and fail loudly
        # instead of silently resampling. On Windows we open WASAPI
        # exclusive. On other platforms the flag is a no-op.
        extra_settings = None
        if self._exclusive_mode:
            try:
                if sys.platform == "darwin":
                    extra_settings = sd.CoreAudioSettings(
                        change_device_parameters=True,
                        fail_if_conversion_required=True,
                    )
                elif sys.platform.startswith("win"):
                    extra_settings = sd.WasapiSettings(exclusive=True)
            except AttributeError:
                # Older sounddevice / PortAudio without these helpers.
                # Fall through to the shared-mode stream so the user
                # still gets audio, just without the exclusive guarantee.
                extra_settings = None

        stream_kwargs: dict = dict(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            device=device,
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
            # PortAudio defaults to its lowest-latency setting (~5-10ms
            # on macOS, similar on Windows WASAPI), which leaves the
            # callback no headroom when a concurrent thread holds the
            # GIL for longer than the buffer can drain. Heavy Python
            # work elsewhere in the app — building a few hundred dicts
            # for the artist endpoint, parsing a 200KB Spotify GraphQL
            # response, occasional gen-0 GC sweeps — can hold the GIL
            # for 5-15ms, and the audio callback (which has to acquire
            # the GIL to enter Python) misses its deadline. The user
            # hears that as stutter.
            #
            # The floor (100ms) gives the callback ~10x more headroom
            # than PortAudio's default; the user-perceived cost of 100ms
            # on play / seek / pause is imperceptible for music. But a
            # fixed 100ms starves a Bluetooth device, whose A2DP pipeline
            # buffers far deeper — that was a continuous-crackle bug on BT
            # output. _output_latency_for honours the device's own deeper
            # latency when it has one, so BT gets the buffer it needs.
            latency=self._output_latency_for(device),
            # Fewer, larger callbacks so sustained GIL contention can't
            # starve the stream; see _CALLBACK_BLOCK_TARGET_S.
            blocksize=self._callback_block_frames(sample_rate),
        )
        if extra_settings is not None:
            stream_kwargs["extra_settings"] = extra_settings

        try:
            self._stream = self._pa_open_stream(stream_kwargs)
        except Exception as open_exc:
            opened = False
            if decoder is not None and stream_kwargs.get("channels", 2) > 2:
                # The device advertised the channels but still refused the
                # open (common on Windows hosts whose mixer is configured
                # for stereo while the jack reports 5.1). Downmix and
                # retry before touching exclusive mode.
                log.warning(
                    "multichannel stream open failed (%s); downmixing to stereo",
                    open_exc,
                )
                decoder.set_target_channels(2)
                stream_kwargs["samplerate"] = decoder.output_sample_rate
                stream_kwargs["channels"] = decoder.channels
                stream_kwargs["dtype"] = decoder.sounddevice_dtype
                try:
                    self._stream = self._pa_open_stream(stream_kwargs)
                    opened = True
                except Exception:
                    pass
            if not opened:
                # Exclusive Mode can fail on devices that refuse the
                # requested rate / format (e.g. a USB DAC pinned to 48k
                # when the track is 44.1k). Fall back to shared mode so
                # playback keeps working. We also need to switch the
                # decoder to mixer-rate output for the same intersample-
                # peak reason as above, since the fallback puts us in
                # shared mode.
                if extra_settings is not None:
                    log.warning(
                        "Exclusive-mode stream open failed; falling back to shared"
                    )
                    stream_kwargs.pop("extra_settings", None)
                    if decoder is not None:
                        fallback_rate = self._query_device_mixer_rate(
                            device, decoder.sample_rate
                        )
                        decoder.set_target_rate(
                            fallback_rate,
                            float_output=self._float_output_for(False),
                        )
                        stream_kwargs["samplerate"] = decoder.output_sample_rate
                        stream_kwargs["dtype"] = decoder.sounddevice_dtype
                        stream_kwargs["channels"] = decoder.channels
                        stream_kwargs["blocksize"] = self._callback_block_frames(
                            decoder.output_sample_rate
                        )
                    self._stream = self._pa_open_stream(stream_kwargs)
                else:
                    raise
        sample_rate = stream_kwargs["samplerate"]
        channels = stream_kwargs["channels"]
        dtype = stream_kwargs["dtype"]

        # Pin the cached stream-state to whatever we actually opened
        # at. set_exclusive_mode and set_output_device read these to
        # decide gapless compatibility, so the values must reflect
        # the post-reconfig rate, not the source rate. Doing this
        # here (vs. at every callsite) keeps the rule in one place.
        self._stream_sample_rate = sample_rate
        self._stream_sd_dtype = dtype
        self._stream_channels = channels

        # Diagnostic line for the audio open. Shows the exact rate /
        # depth / channel count / host API / exclusive flag we ended
        # up using. Critical for triaging Windows-side audio bugs
        # where a stutter or freeze typically traces back to a rate
        # mismatch between the decoder and the device, or to
        # exclusive-mode being on for a device that only supports a
        # subset of the rates we're feeding. Printed unconditionally
        # so it lands in the user's console log without needing a
        # debug flag — the volume is one line per stream open.
        try:
            dev_info = sd.query_devices(device, kind="output")
            ha = sd.query_hostapis(dev_info.get("hostapi"))
            ha_name = ha.get("name") if isinstance(ha, dict) else "?"
        except Exception:
            ha_name = "?"
        source_channels = (
            decoder.source_channels if decoder is not None else None
        )
        print(
            f"[audio] stream open: device={device} rate={sample_rate}Hz "
            f"channels={channels} source_channels={source_channels} "
            f"device_max_channels={device_max} dtype={dtype!r} "
            f"hostapi={ha_name!r} exclusive={self._exclusive_mode} "
            f"block={stream_kwargs['blocksize']}",
            file=sys.stderr,
            flush=True,
        )
        # Mirror to the persistent audio log too: a Finder-launched
        # build has no console, and whether a 5.1/7.1 bed actually
        # reached the endpoint is the first thing to check when
        # spatial audio "doesn't work".
        audio_log.info(
            "stream open: device=%s rate=%sHz channels=%s "
            "source_channels=%s device_max_channels=%s dtype=%r "
            "hostapi=%r exclusive=%s",
            device,
            sample_rate,
            channels,
            source_channels,
            device_max,
            dtype,
            ha_name,
            self._exclusive_mode,
        )
        # Remember what this open actually negotiated. The Signal Path
        # panel reads the live stream fields; the Atmos tester needs
        # this after the fact — a native/PCM A-B comparison is only
        # meaningful with the host API and channel count attached.
        with self._lock:
            self._last_open_diag = {
                "device": device,
                "device_name": dev_info.get("name"),
                "hostapi": ha_name,
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "source_channels": source_channels,
                "device_max_channels": device_max,
                "dtype": dtype,
                "exclusive": bool(self._exclusive_mode),
            }

        # Rebuild the EQ against the new sample rate. Preserves
        # whichever mode is active — manual bands or AutoEQ profile.
        self._eq = Equalizer(sample_rate=sample_rate, channels=channels)
        if self._eq_profile is not None:
            try:
                from app.audio.autoeq.apply import (
                    TiltConfig,
                    cascade_with_tilt,
                )

                tilt = self._eq_tilt or TiltConfig()
                sos, preamp_db = cascade_with_tilt(
                    self._eq_profile, sample_rate, tilt
                )
                if sos.size == 0:
                    self._eq.clear()
                else:
                    self._eq.set_sos(sos, preamp_db=preamp_db)
            except Exception:
                log.exception("autoeq profile coefficient build failed")
                self._eq.clear()
        elif self._eq_parametric_bands or self._eq_preamp is not None:
            # `or preamp` — a preamp-only curve (no bands) must
            # survive the reopen too; set_sos installs the
            # preamp-only stage for an empty cascade.
            try:
                sos = build_parametric_sos(
                    self._eq_parametric_bands, sample_rate
                )
                self._eq.set_sos(sos, preamp_db=self._eq_preamp)
            except Exception:
                log.exception("eq coefficient build failed")
                self._eq.clear()

        # Crossfeed coefficients depend on the active sample rate, so
        # rebuild it here too. The user setting (`_crossfeed_amount`)
        # survives the rebuild — same pattern as EQ.
        self._crossfeed = Crossfeed(sample_rate=sample_rate)
        if self._crossfeed_amount > 0:
            try:
                self._crossfeed.set_amount(self._crossfeed_amount)
            except Exception:
                log.exception("crossfeed coefficient build failed")
                self._crossfeed.clear()

    @staticmethod
    def _device_max_output_channels(device: Optional[int]) -> Optional[int]:
        """How many output channels the device's current mixer config
        exposes, or None when PortAudio can't tell us."""
        try:
            info = sd.query_devices(device, kind="output")
        except Exception:
            return None
        value = info.get("max_output_channels") if isinstance(info, dict) else None
        try:
            channels = int(value)
        except (TypeError, ValueError):
            return None
        return channels if channels > 0 else None

    @staticmethod
    def _query_device_mixer_rate(device: Optional[int], fallback: int) -> int:
        """Look up the rate the OS will mix at for `device` in shared
        mode. If sounddevice can't tell us, return `fallback` so the
        decoder stays at source rate (the previous behavior). Rounded
        to int because PortAudio reports it as a float."""
        try:
            info = sd.query_devices(device, kind="output")
        except Exception:
            log.warning("query_devices failed; using source rate as fallback")
            return fallback
        rate = info.get("default_samplerate") if isinstance(info, dict) else None
        if not rate or rate <= 0:
            return fallback
        return int(round(rate))

    # Floor on the requested output latency. This is the headroom the
    # realtime callback needs to ride out GIL/CPU contention on wired,
    # low-latency devices — see the long note in _open_output_stream.
    _OUTPUT_LATENCY_FLOOR_S = 0.1

    # How much audio one callback delivers. The latency floor above
    # absorbs a single long stall, but it does nothing for sustained
    # contention, because what decides that is how often the callback
    # has to win the GIL. Left to PortAudio, the callback runs once per
    # device period (10 ms on WASAPI shared), and each run must
    # reacquire the GIL before it copies a sample. With two or three
    # request threads running Python, that handoff averages more than
    # 10 ms (on Windows the waiter's timed wait rounds up to the 15.6 ms
    # system timer tick), so the callback falls further behind every
    # period and WASAPI plays silence in the gaps. PortAudio does not
    # flag those as underflows, so output_underruns stays 0 while half
    # the audio is missing. Measured against the real player on WASAPI
    # shared: with two busy threads, 10 ms callbacks delivered 72% of
    # the audio and 46 ms callbacks 99.9%; with three, 51% vs 95%. The
    # cost is about one block of added latency on pause / volume / seek
    # (~110 ms -> ~150 ms total).
    _CALLBACK_BLOCK_TARGET_S = 0.045

    @staticmethod
    def _callback_block_frames(sample_rate: int) -> int:
        """Frames per audio callback for `sample_rate`: the power of two
        nearest _CALLBACK_BLOCK_TARGET_S (2048 at 44.1 / 48 kHz, 4096 at
        88.2 / 96 kHz, 8192 at 176.4 / 192 kHz)."""
        target = max(1.0, sample_rate * PCMPlayer._CALLBACK_BLOCK_TARGET_S)
        return 1 << round(math.log2(target))

    @staticmethod
    def _float_output_for(exclusive: bool) -> bool:
        """Whether decoders feeding a stream should emit float32 even at
        the source rate.

        True for shared-mode streams on Windows and macOS. WASAPI's and
        CoreAudio's mixers run in float32, so integer output buys no
        bit-perfectness there: PortAudio converts it to float32 in C,
        exactly, before the mixer sees it. What integer output does cost
        is in the realtime callback. Its volume / ReplayGain / EQ stage
        then casts int -> float -> int, and numpy releases the GIL during
        those casts, so every callback has to win the GIL back from
        whatever request thread took it. With three busy threads that
        stage took 12 ms median and 118 ms worst case per 46 ms block;
        on float32 it runs in place in about 10 us under the same load.

        Exclusive mode keeps the source format, which is its point.
        Linux is left alone: PortAudio can reach an ALSA hw device
        directly there, where integer output can still matter."""
        return not exclusive and sys.platform in ("win32", "darwin")

    @staticmethod
    def _output_latency_for(device: Optional[int]) -> float:
        """Suggested PortAudio output latency for `device`.

        The floor (_OUTPUT_LATENCY_FLOOR_S) is what wired/low-latency
        outputs need for GIL headroom. Bluetooth is different: its A2DP
        pipeline buffers deeply (~150-300 ms), so asking PortAudio for
        only the floor starves the device and the user hears continuous
        crackle on BT output. A device's own `default_high_output_latency`
        reflects that real pipeline depth, so honour whichever is larger.
        `device=None` resolves to the default output. Best-effort: any
        query failure falls back to the floor (the previous fixed value)."""
        try:
            info = sd.query_devices(device, kind="output")
        except Exception:
            return PCMPlayer._OUTPUT_LATENCY_FLOOR_S
        high = info.get("default_high_output_latency") if isinstance(info, dict) else None
        try:
            high_f = float(high)
        except (TypeError, ValueError):
            high_f = 0.0
        return max(PCMPlayer._OUTPUT_LATENCY_FLOOR_S, high_f)

    def _ensure_stream_started(self) -> None:
        """Start the open output stream, reopening it if the first
        attempt fails.

        PortAudio's Windows backend raises `Unanticipated host error
        [PaErrorCode -9999]` from IAudioClient::Start when the driver
        hasn't finished releasing the endpoint after close() — the
        gapless swap in load() exists to avoid exactly this race, but
        a full teardown + fresh open (format change, device change,
        stop) still has to close and reopen. When that race is lost,
        every following play fails the same way until the endpoint
        settles, so reopening after a short delay instead of parking
        the player in `error` is what keeps the queue moving.

        A Windows Spatial Sound change ("Dolby Atmos for Headphones"
        is the common one) is the harder case: the endpoint is torn
        down and rebuilt, and the cached WASAPI copy can then refuse
        to start for as long as PortAudio keeps its stale enumeration
        — observed as a minute of failures while a fresh process
        opened the same device fine. So the retries widen each round:
        reopen, re-enumerate PortAudio, route to another host-API copy
        of the same speakers (MME/DirectSound are untouched by the
        spatial APO), then the system default. The last attempt's
        exception is re-raised so a genuinely dead device still
        surfaces in the UI.
        """
        stream = self._stream
        if stream is None or stream.active:
            return
        # WASAPI needs COM on the thread that starts the stream, and
        # this runs on HTTP workers / recovery threads (win_com.py).
        try:
            self._start_stream(stream)
            return
        except Exception as exc:
            log.warning("stream.start failed (%s); reopening", exc)
            audio_log.warning("stream.start failed (%s); reopening", exc)

        last_exc: Optional[BaseException] = None
        # Muzzle the finished_callback the aborted stream fires below:
        # with the device-loss heuristic a stop from a stream we are
        # replacing must not spawn a second recovery on top of this one.
        with self._lock:
            self._replacing_stream = True
        try:
            for delay, tier in (
                (0.3, "reopen"),
                (0.7, "reenumerate"),
                (1.5, "alternate-hostapi"),
                (2.0, "system-default"),
            ):
                time.sleep(delay)
                self._close_failed_stream()
                if tier == "reenumerate":
                    self._reinit_portaudio()
                try:
                    self._open_stream_tier(tier)
                    audio_log.info(
                        "stream.start recovered after reopen (%s)", tier
                    )
                    print(
                        f"[audio] stream.start recovered after reopen "
                        f"({tier})",
                        flush=True,
                    )
                    return
                except Exception as exc:
                    last_exc = exc
                    audio_log.warning(
                        "stream.start reopen failed (%s): %s", tier, exc
                    )
            # Every tier failed: drop the corpse so the next load takes
            # the full-reopen path instead of rebinding a new track to
            # a stream that never started. That reuse is what turned
            # one bad endpoint into a queue-wide track-advance loop.
            self._close_failed_stream()
        finally:
            with self._lock:
                self._replacing_stream = False
        if last_exc is not None:
            raise last_exc

    def _close_failed_stream(self) -> None:
        old = self._stream
        self._stream = None
        self._callback_carry = None
        _close_pa_stream(old)

    @staticmethod
    def _reinit_portaudio() -> None:
        """Force PortAudio to re-enumerate the endpoints.

        The device list — and with it each WASAPI endpoint's state — is
        cached at init time. A spatial-audio change invalidates the
        cached endpoint in place, so a reopen without this keeps
        hitting the same dead client until the process restarts.

        pa_guard refuses while any stream in the process is still open
        (ours, or the Atmos tester's own player): terminating PortAudio
        under a live handle crashes the next call on it. The caller
        then walks on to the next recovery tier, which works without a
        fresh enumeration.
        """
        pa_guard.reinit_portaudio()

    def _open_stream_tier(self, tier: str) -> None:
        """One recovery attempt of `_ensure_stream_started`. Raises on
        failure so the caller can walk to the next tier."""
        sample_rate = self._stream_sample_rate or 48000
        channels = self._stream_channels or 2
        sd_dtype = self._stream_sd_dtype or "float32"
        if tier == "alternate-hostapi":
            current, _ = resolve_output_device(self._selected_device_id)
            alt, alt_hostapi = resolve_fallback_output_device(
                self._selected_device_id,
                channels,
                sample_rate,
                sd_dtype,
                exclude=current,
            )
            if alt is None:
                raise RuntimeError(
                    "no other host-API copy of the device can open"
                )
            log.info(
                "stream recovery: routing to device %d (hostapi=%s)",
                alt,
                alt_hostapi,
            )
            print(
                f"[audio] stream recovery: routing to device {alt} "
                f"hostapi={alt_hostapi!r}",
                flush=True,
            )
            self._open_output_stream(
                sample_rate, channels, sd_dtype, device_override=alt
            )
        elif tier == "system-default":
            saved = self._selected_device_id
            self._selected_device_id = ""
            try:
                self._open_output_stream(sample_rate, channels, sd_dtype)
            finally:
                self._selected_device_id = saved
        else:
            self._open_output_stream(sample_rate, channels, sd_dtype)
        fresh = self._stream
        if fresh is None:
            raise RuntimeError("output stream open produced no stream")
        fresh.start()

    def _start_decoder_thread(self) -> None:
        # Bind the decoder / queue / flags to the thread's locals at
        # creation time. The loop must NOT read self.* fields — a
        # concurrent load() may rebind those, which previously caused
        # two decoder threads to share one PyAV generator (hence
        # "generator already executing" errors under rapid back-to-
        # back play_track calls).
        decoder = self._decoder
        pcm_queue = self._pcm_queue
        stop_flag = self._stop_flag
        done_event = self._decoder_done
        t = threading.Thread(
            target=self._decoder_loop,
            args=(decoder, pcm_queue, stop_flag, done_event),
            name="pcm-decoder",
            daemon=True,
        )
        self._decoder_thread = t
        t.start()

    @staticmethod
    def _decoder_loop(
        decoder: Optional[Decoder],
        pcm_queue: "queue.Queue[Optional[np.ndarray]]",
        stop_flag: threading.Event,
        done_event: threading.Event,
    ) -> None:
        if decoder is None:
            done_event.set()
            return
        try:
            while not stop_flag.is_set():
                pcm = decoder.next_pcm()
                if pcm is None:
                    break
                while not stop_flag.is_set():
                    try:
                        pcm_queue.put(pcm, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception:
            log.exception("decoder thread crashed")
        finally:
            done_event.set()
            # Push the sentinel onto the thread's OWN queue, not
            # whatever self._pcm_queue happens to point at now. If
            # a gapless swap rebound self._pcm_queue to a different
            # queue mid-loop, pushing the sentinel there would have
            # incorrectly flagged the new decoder as done.
            try:
                pcm_queue.put_nowait(None)
            except queue.Full:
                pass

    def audio_health(self) -> dict:
        """Cumulative playback-health counters since this player was
        constructed, for the activity report. Each count maps to a
        distinct stutter cause so a bug report can be triaged without
        the audio.log:

          output_underruns  — PortAudio flagged an over/underrun
                              (driver / exclusive-mode / device can't
                              keep up).
          queue_starvations — our decoder/network didn't fill the PCM
                              buffer in time (throughput / CPU).
          callback_jitter_events — the realtime callback was delivered
                              late while the queue was full (GIL/CPU
                              contention elsewhere in the app).

        worst_jitter_late_ms is the largest single late delivery seen.
        The queue depth/max and samples_emitted give a point-in-time
        sense of buffer fill and how much has played. All best-effort:
        the queue may not exist yet if nothing has played.
        """
        q = getattr(self, "_pcm_queue", None)
        try:
            queue_depth = q.qsize() if q is not None else None
        except Exception:
            queue_depth = None
        return {
            "output_underruns": self._cb_status_total,
            "queue_starvations": self._cb_starve_total,
            "callback_jitter_events": self._cb_jitter_total,
            "worst_jitter_late_ms": round(self._cb_jitter_worst_late_ms, 1),
            "samples_emitted": getattr(self, "_samples_emitted", 0),
            "pcm_queue_depth": queue_depth,
            "pcm_queue_max": _PCM_QUEUE_MAX,
        }

    def _log_callback_status(self, status) -> None:
        """Rate-limited diagnostic for PortAudio over/underruns
        signalled via the callback's `status` arg. One message + a
        running count per second; without rate limiting a sustained
        glitch at 86 Hz callback cadence would saturate stderr.
        Critical diagnostic on Windows where stutter complaints
        typically trace back to this path."""
        now = time.monotonic()
        self._cb_status_count += 1
        self._cb_status_total += 1
        if now - self._cb_status_last_print >= 1.0:
            count = self._cb_status_count
            self._cb_status_count = 0
            self._cb_status_last_print = now
            print(
                f"[audio] callback status={status} "
                f"(count_since_last={count})",
                file=sys.stderr,
                flush=True,
            )
            try:
                audio_log.info(
                    "callback status=%s count_since_last=%d", status, count
                )
            except Exception:
                pass

    def _log_callback_heartbeat(self) -> None:
        """DEBUG-only per-100-callback heartbeat. Confirms the
        callback is running + advancing samples and shows the
        queue depth so a "decoder is filling but callback isn't
        draining" condition is visible at a glance."""
        self._callback_counter += 1
        if self._callback_counter % 100 == 0:
            qsize = self._pcm_queue.qsize()
            print(
                f"[pcm] callback tick #{self._callback_counter} "
                f"samples_emitted={self._samples_emitted} "
                f"queue={qsize}/{_PCM_QUEUE_MAX} "
                f"state={self._state} "
                f"track={self._current_track_id}",
                file=sys.stderr, flush=True,
            )

    def _log_callback_starvation(self) -> None:
        """Rate-limited diagnostic for our-side underruns — the
        decoder didn't push a chunk in time. Distinct from
        `_log_callback_status` because it pinpoints the cause as
        ours (slow disk / network / CPU) rather than driver-side."""
        now = time.monotonic()
        self._cb_starve_count += 1
        self._cb_starve_total += 1
        if now - self._cb_starve_last_print >= 1.0:
            count = self._cb_starve_count
            self._cb_starve_count = 0
            self._cb_starve_last_print = now
            print(
                f"[audio] queue starvation: decoder behind "
                f"the callback (count_since_last={count})",
                file=sys.stderr,
                flush=True,
            )
            try:
                audio_log.info(
                    "queue starvation count_since_last=%d", count
                )
            except Exception:
                pass

    def _log_callback_jitter(self, gap_s: float, expected_s: float) -> None:
        """Rate-limited diagnostic for a missed callback deadline: the
        realtime thread was serviced `gap_s` after the previous audio
        callback, well past the `expected_s` buffer period. Distinct
        from starvation (queue had data) and from PortAudio status
        (the device didn't necessarily flag it) — this is the GIL/CPU
        contention crackle, the kind you hear when navigating pages
        mid-playback. One line + worst-case + count per second."""
        now = time.monotonic()
        self._cb_jitter_count += 1
        self._cb_jitter_total += 1
        gap_ms = gap_s * 1000.0
        if gap_ms > self._cb_jitter_worst_ms:
            self._cb_jitter_worst_ms = gap_ms
        late_ms = gap_ms - expected_s * 1000.0
        if late_ms > self._cb_jitter_worst_late_ms:
            self._cb_jitter_worst_late_ms = late_ms
        if now - self._cb_jitter_last_print >= 1.0:
            count = self._cb_jitter_count
            worst = self._cb_jitter_worst_ms
            self._cb_jitter_count = 0
            self._cb_jitter_worst_ms = 0.0
            self._cb_jitter_last_print = now
            msg = (
                f"callback jitter: late by "
                f"{gap_ms - expected_s * 1000.0:.0f}ms "
                f"(gap={gap_ms:.0f}ms expected={expected_s * 1000.0:.1f}ms "
                f"worst={worst:.0f}ms count_since_last={count}) "
                f"— missed audio deadline, queue was not starved"
            )
            print(f"[audio] {msg}", file=sys.stderr, flush=True)
            try:
                audio_log.info(msg)
            except Exception:
                pass

    def _stall_watchdog(self) -> None:
        """Detect a wedged audio callback and dump every thread's
        stack so the freeze diagnoses itself.

        The realtime callback sets _cb_last_t on every invocation,
        including while paused/seeking (it still fires to output
        silence), so a stale _cb_last_t while state is "playing"
        means the callback thread is blocked, not idle. 10s is far
        beyond any normal buffer/jitter (callbacks run ~12x/sec), so
        there are no false positives from scheduling hiccups.

        Strictly lock-free: it must be able to fire *during* a _lock
        deadlock, so it only reads plain attributes and never takes
        _lock. Fires once per stall; rearms when audio resumes.
        """
        STALL_S = 10.0
        while True:
            time.sleep(5.0)
            try:
                self._tick_load_stall()
            except Exception:
                pass
            try:
                last = self._cb_last_t
                active = (
                    self._stream is not None
                    and self._state == "playing"
                    and not self._paused
                    and not self._seeking
                    and bool(last)
                )
                if not active:
                    self._stall_dumped = False
                    continue
                stream = self._stream
                if self._watchdog_recover_dead_stream(stream):
                    self._stall_dumped = False
                    continue
                silent_for = time.monotonic() - last
                if silent_for < STALL_S:
                    self._stall_dumped = False
                    continue
                if self._stall_dumped:
                    continue
                self._stall_dumped = True
                from app.paths import user_data_dir

                path = user_data_dir() / "audio_stall.txt"
                header = (
                    f"AUDIO STALL: callback silent for {silent_for:.1f}s "
                    f"state={self._state} track={self._current_track_id}"
                )
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(header + "\n\n")
                        faulthandler.dump_traceback(fh, all_threads=True)
                except Exception:
                    pass
                try:
                    audio_log.info(f"{header}; thread dump -> {path}")
                except Exception:
                    pass
                print(
                    f"[player] {header}; thread dump -> {path}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                # The watchdog must never take the process down.
                pass

    def _watchdog_recover_dead_stream(self, stream) -> bool:
        """Kick device-loss recovery when the output stream is gone
        while the player reports "playing".

        A Windows Spatial Sound change reconfigures the endpoint in
        place, and the old client just disappears; PortAudio does not
        reliably call finished_callback for that, so without this the
        player would sit in "playing" over silence until the user
        noticed. Recovery reopens at the same position — the decoder
        queue is untouched. Rate limited because the watchdog ticks
        every 5s and a reopen takes a moment.

        Lock-free, like its caller: plain attribute reads only.
        """
        if stream is None or stream.active or self._replacing_stream:
            return False
        now = time.monotonic()
        if now - self._last_watchdog_recover_t < 10.0:
            return False
        self._last_watchdog_recover_t = now
        try:
            audio_log.warning(
                "watchdog: output stream inactive while playing "
                "(device reconfigured?); recovering"
            )
        except Exception:
            pass
        threading.Thread(
            target=self._recover_from_device_loss,
            name="pcm-watchdog-recovery",
            daemon=True,
        ).start()
        return True

    def _tick_load_stall(self) -> None:
        """One stuck-loading recovery tick.

        A load that never produces audio leaves the player pinned in
        "loading" forever, so past _LOAD_STALL_S we surface an error and
        the frontend (with "continue playing" on) auto-advances past the
        dead track.

        The DLNA cast file wait is excluded: it is bounded by
        _CAST_FILE_READY_S and is *expected* to take longer than
        _LOAD_STALL_S for long, hi-res tracks. Counting it here forced a
        false error that the frontend turned into a track-advance loop —
        every new track stalled in the same bounded wait and errored
        again. Dropping the timer during that wait gives the portion
        after it a fresh window. Strictly lock-free, like the watchdog
        that drives it."""
        if self._state != "loading":
            self._loading_since = None
            return
        if self._awaiting_cast_file:
            self._loading_since = None
            return
        if self._loading_since is None:
            self._loading_since = time.monotonic()
            return
        if time.monotonic() - self._loading_since > _LOAD_STALL_S:
            self._force_load_stall_error()
            self._loading_since = None

    def _force_load_stall_error(self) -> None:
        """Recover a load wedged in "loading" past LOAD_STALL_S by
        surfacing an error, so the UI stops hanging and (with continue-
        playing on) the frontend advances past the dead track.

        Uses a NON-blocking lock acquire to keep the watchdog's
        survive-a-deadlock invariant: if `_lock` is contended we skip
        this round rather than park the watchdog. Best-effort and never
        raises out to the caller."""
        track = self._current_track_id
        if not self._lock.acquire(blocking=False):
            return
        try:
            # Re-check under the lock — the load may have just completed.
            if self._state != "loading":
                return
            self._state = "error"
            self._last_error = "Track load timed out"
            self._seq += 1
        finally:
            self._lock.release()
        print(
            f"[pcm] load-stall watchdog: track={track} stuck in 'loading' "
            f">30s; forced error so playback can recover",
            file=sys.stderr,
            flush=True,
        )
        try:
            audio_log.info(
                "load-stall: track=%s stuck in loading; forced error", track
            )
        except Exception:
            pass
        try:
            self._emit()
        except Exception:
            pass

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self._log_callback_status(status)
        if PCMPlayer._DEBUG:
            self._log_callback_heartbeat()

        # Callback-jitter check. Only meaningful between two
        # consecutive callbacks that both delivered continuous audio;
        # a gap after pause/seek/swap/underrun/stop is expected, not a
        # glitch, so those paths leave _cb_prev_audio False. A gap of
        # >= 2x the buffer period means the realtime thread was
        # serviced a full buffer late — that's the audible crackle.
        now = time.monotonic()
        if self._cb_prev_audio and self._cb_last_t:
            rate = self._stream_sample_rate or 0
            if rate:
                expected = frames / float(rate)
                gap = now - self._cb_last_t
                if expected > 0.0 and gap >= expected * 2.0:
                    self._log_callback_jitter(gap, expected)
        self._cb_last_t = now
        self._cb_prev_audio = False

        # Paused → output silence, don't drain queue, don't advance
        # position. Zero-latency resume: on unpause the next call
        # finds the queue exactly where it was.
        if self._paused:
            outdata.fill(0)
            return

        # Seeking → also output silence. Without this, the brief
        # window where the old decoder has exited and the new one
        # hasn't started would let the callback raise CallbackStop
        # and end the stream mid-seek.
        if self._seeking:
            outdata.fill(0)
            return

        # A pipeline swap is rewriting decoder/queue/gain refs on
        # another thread. This callback is lock-free, so reading a
        # half-swapped pipeline would feed the new track's samples
        # through the old track's ReplayGain/filters — the
        # split-second "explosion" on track change. Silence until the
        # swap (including the ReplayGain re-derive) completes; it's
        # sub-millisecond, so this is inaudible.
        if self._swapping:
            outdata.fill(0)
            return

        written = 0
        channels = outdata.shape[1]

        # Crossfade: when enabled and the current track is within the fade
        # window, mix the next track in instead of cutting to it. Fully
        # self-contained — when off (the default) or any precondition is
        # unmet, `_xfade_active` stays False and the normal single-queue
        # loop below runs unchanged. Setting `written = frames` skips that
        # loop; the shared output stages (cast/EQ/volume) and the
        # samples_emitted / seq bump below then apply to the mix exactly as
        # they do to normal audio.
        if not self._xfade_active and self._crossfade_s > 0.0:
            self._maybe_begin_crossfade()
        if self._xfade_active:
            self._crossfade_fill(outdata, frames, channels)
            written = frames

        while written < frames:
            # Re-check per iteration, not just at the top of the
            # callback. `seek()` sets `_seeking` and then clears
            # `_callback_carry` from the HTTP thread; a callback that
            # already passed the guard above is inside this loop by
            # then and would keep feeding the pre-seek buffer. Yielding
            # here is what the guard intends — the next callback starts
            # clean on the new decoder.
            if self._seeking or self._swapping:
                outdata[written:] = 0
                return
            if self._callback_carry is None:
                try:
                    chunk = self._pcm_queue.get_nowait()
                except queue.Empty:
                    chunk = None
                if chunk is None:
                    if self._decoder_done.is_set():
                        # Primary track exhausted. If a compatible
                        # preload is waiting, splice it into the
                        # SAME OutputStream — that's the gapless
                        # moment. Re-enter the loop with the new
                        # queue.
                        if self._try_gapless_swap():
                            continue
                        outdata[written:] = 0
                        raise sd.CallbackStop
                    # Underrun — decoder hasn't caught up. Silence
                    # the rest of this callback. Critically we still
                    # advance samples_emitted below so the scrubber
                    # keeps moving through the underrun window —
                    # otherwise position would freeze on every
                    # hiccup. Bump seq on the same wall-clock cadence
                    # as the main path so the frontend's seq-dedup
                    # actually lets the position update through.
                    self._log_callback_starvation()
                    outdata[written:] = 0
                    self._samples_emitted += frames
                    if now - self._seq_last_bump_t >= 0.2:
                        self._seq += 1
                        self._seq_last_bump_t = now
                    return
                self._callback_carry = chunk

            # Read once into a local and use it for the whole iteration.
            # This callback is lock-free, so `seek()` clearing
            # `_callback_carry` between the assignment above and the
            # reads below turned it into None mid-expression and raised
            # "'NoneType' object has no attribute 'shape'", killing the
            # callback and stalling audio until the stream was torn
            # down (#405). A local can't be nulled underneath us.
            carry = self._callback_carry
            if carry is None:
                # Cleared between the check and here — treat it as a
                # seek in flight rather than reaching for .shape.
                outdata[written:] = 0
                return

            take = min(frames - written, carry.shape[0])
            src = carry[:take]
            if src.shape[1] == channels:
                outdata[written : written + take] = src
            elif src.shape[1] == 2 and channels == 1:
                outdata[written : written + take, 0] = src.mean(axis=1)
            else:
                outdata[written : written + take] = 0
            written += take
            if take >= carry.shape[0]:
                self._callback_carry = None
            else:
                self._callback_carry = carry[take:]

        # Cast tap. Runs before EQ / volume / mute so the device
        # gets the raw decoded PCM; the Cast device has its own
        # volume control, so muting
        # locally shouldn't silence the remote speaker. The
        # is_active() probe is lock-free in the common 'no
        # session' case, so the cost when nobody's casting is one
        # attribute read per audio callback. Tidal sources decode
        # to int16 or int32, but on Windows WASAPI shared mode the
        # OutputStream comes out as float32 (device mixer format),
        # so we hand all three through; cast_manager.push_pcm
        # converts float32 to int32 internally for the FLAC encoder.
        if _cast_mod is not None:
            try:
                if _cast_mod.cast_manager.is_active():
                    if outdata.dtype == np.int16:
                        _dtype_name = "int16"
                    elif outdata.dtype == np.int32:
                        _dtype_name = "int32"
                    elif outdata.dtype == np.float32:
                        _dtype_name = "float32"
                    else:
                        _dtype_name = None
                    if _dtype_name is not None:
                        _cast_mod.cast_manager.push_pcm(
                            np.ascontiguousarray(outdata),
                            sample_rate=self._stream_sample_rate or 44100,
                            dtype=_dtype_name,
                        )
            except Exception:
                # Never let Cast errors take down local playback.
                pass

        # UPnP / DLNA tap. Same pre-EQ / pre-volume position and
        # same dtype handling as Cast. The streaming pipeline is
        # FLAC-over-HTTP for both, so the encoder ingestion contract
        # is identical. `is_active()` is a single attribute read,
        # and `upnp_manager` is a module-level singleton (no lock,
        # no lazy-init branch), so the cost when DLNA isn't in use
        # is exactly two attribute reads per audio callback.
        if _upnp_manager is not None:
            try:
                if _upnp_manager.is_active():
                    if outdata.dtype == np.int16:
                        _dtype_name = "int16"
                    elif outdata.dtype == np.int32:
                        _dtype_name = "int32"
                    elif outdata.dtype == np.float32:
                        _dtype_name = "float32"
                    else:
                        _dtype_name = None
                    if _dtype_name is not None:
                        _upnp_manager.push_pcm(
                            np.ascontiguousarray(outdata),
                            sample_rate=self._stream_sample_rate or 44100,
                            dtype=_dtype_name,
                        )
            except Exception:
                # Never let DLNA errors take down local playback.
                pass

        # External output active: silence local. Done AFTER the
        # Cast / Tidal Connect taps above (so the remote
        # receiver gets full-amplitude audio at its own volume
        # control) and BEFORE the volume / mute logic below (so the
        # silencing is unconditional regardless of user volume
        # state). The OutputStream still runs — we just hand it
        # zeros — which keeps the realtime callback driving and
        # avoids the underrun-recovery dance that stopping +
        # restarting would cost when the user toggles back to
        # local.
        if self._external_output_active:
            outdata.fill(0)
            self._samples_emitted += frames
            # Bump seq so the frontend's position scrubber still
            # advances even though local is silent — playback is
            # still happening, just on the remote.
            if now - self._seq_last_bump_t >= 0.2:
                self._seq += 1
                self._seq_last_bump_t = now
            return

        # DSP stages: ReplayGain (scalar gain) → EQ (10-band biquad
        # / AutoEQ) → crossfeed (Bauer stereo imaging). All three
        # operate on float32, so when any of them is active we
        # round-trip through a single float32 buffer instead of
        # paying the int↔float conversion per stage. Bit-perfect
        # pass-through holds when all three are inactive and we
        # never enter this block.
        #
        # RG goes first because it sets the base level the rest of
        # the chain shapes around — applying it after EQ would let
        # an EQ-boosted band push the post-RG peak into clipping.
        rg_active = self._replaygain.is_active()
        eq_active = (
            self._eq is not None
            and self._eq.is_active()
            and not self._eq_bypass
        )
        # Crossfeed only makes sense for stereo; mono / 5.1 pay
        # nothing for the feature being on.
        crossfeed_active = (
            self._crossfeed is not None
            and self._crossfeed.is_active()
            and outdata.ndim == 2
            and outdata.shape[1] == 2
        )
        if rg_active or eq_active or crossfeed_active:
            if outdata.dtype == np.float32:
                if rg_active:
                    self._replaygain.apply(outdata)
                if eq_active:
                    self._eq.apply(outdata)
                if crossfeed_active:
                    self._crossfeed.apply(outdata)
            else:
                # Scale to float32 in the range [-1, 1], filter,
                # scale back. int range constants chosen so the
                # round-trip is exact for unmodified samples.
                scale_in = (
                    32768.0 if outdata.dtype == np.int16 else 2_147_483_648.0
                )
                buf = outdata.astype(np.float32, copy=True) / scale_in
                if rg_active:
                    self._replaygain.apply(buf)
                if eq_active:
                    self._eq.apply(buf)
                if crossfeed_active:
                    self._crossfeed.apply(buf)
                np.clip(buf * scale_in, -scale_in, scale_in - 1.0, out=buf)
                # np.rint rounds to nearest (banker's); plain astype
                # would truncate toward zero and bias samples slightly
                # negative on average.
                outdata[:] = np.rint(buf).astype(outdata.dtype)

        # Volume + mute post-processing. At volume=100 and not muted
        # (and no EQ above), bit-perfect pass-through still holds when
        # the decoder isn't doing internal SRC; when it is, the
        # decoder has already attenuated by ~1 dB upstream so peaks
        # land below full scale.
        if self._muted or self._volume <= 0:
            outdata.fill(0)
        elif self._volume < 100:
            vol = self._volume / 100.0
            if outdata.dtype == np.float32:
                outdata *= vol
            else:
                # int16 / int32: scale via float roundtrip. The float
                # multiplier is safe for a 32-bit int dtype because
                # we clamp in astype().
                scaled = outdata.astype(np.float32, copy=False) * vol
                outdata[:] = scaled.astype(outdata.dtype)

        self._samples_emitted += frames
        # Reached the end having delivered real audio this callback,
        # so the gap to the NEXT callback is a meaningful deadline
        # measurement (see the jitter check at the top).
        self._cb_prev_audio = True
        # Bump seq so the frontend's SSE dedupe lets through the
        # position update. Wall-clock cadence (5 Hz) — independent of
        # the callback's own rate. PortAudio's frames-per-callback
        # depends on the device + buffer configuration: ~512 on a
        # 44.1k Mac (~86 Hz callback), but as low as ~10 Hz callback
        # for hi-res-output configs that prefer larger buffers.
        # The earlier "/20" scheme tracked the 86 Hz case and
        # silently regressed everywhere else — the frontend's
        # seq-dedup would drop multiple 4 Hz SSE pollings between
        # consecutive seq bumps, so the scrubber jumped multiple
        # seconds at a time instead of sliding smoothly.
        if now - self._seq_last_bump_t >= 0.2:
            self._seq += 1
            self._seq_last_bump_t = now

    def _adopt_preload_locked(self, pre: _Preload) -> PlayerSnapshot:
        """Synchronous version of the callback's gapless swap,
        triggered from load() when the frontend fires play_track
        for a track we already have preloaded.

        Must be called with `self._lock` held. Stops the current
        decoder thread, swaps refs to the preload's pipeline, and
        either keeps the OutputStream alive (same rate/dtype —
        gapless) or re-opens it (different rate — small reopen
        gap). Does not tear the preload's decoder thread: it keeps
        running against its own queue, which now becomes the
        primary queue.
        """
        if getattr(pre, "cast", False) is True:
            # Bounded-DLNA cast: there is no local decoder to swap. The
            # renderer streams the bounded file; just move the backend's
            # current track so the frontend's follow-up play_track(next)
            # lands on Path 0 and the UI re-syncs off the `playing`
            # snapshot. Idempotent: a duplicate adopt of the track we are
            # already on is a no-op.
            if (
                self._current_track_id == pre.track_id
                and self._state == "playing"
            ):
                self._preload = None
                return self.snapshot()
            self._current_track_id = pre.track_id
            self._current_duration_ms = pre.duration_ms
            self._current_stream_info = pre.stream_info
            if pre.track_meta is not None:
                self._current_track_meta = pre.track_meta
            self._source_urls = pre.source_urls
            self._source_path = pre.source_path
            self._preload = None
            self._state = "playing"
            self._last_error = None
            self._seq += 1
            # Promote the pre-staged bounded file to the current source
            # and re-arm the renderer watchdog for the new track. The
            # sticky consumed-track evidence makes start_passthrough's
            # announce a no-op, so the renderer is not Stop/restarted —
            # its clock just continues into the file it already holds.
            if (
                _upnp_manager is not None
                and _upnp_manager.is_active()
                and pre.source_urls is not None
            ):
                try:
                    _upnp_manager.start_passthrough(
                        pre.source_urls,
                        prefetched=None,
                        metadata=pre.track_meta or self._current_track_meta,
                        track_id=pre.track_id,
                    )
                except Exception as exc:  # noqa: BLE001 - passthrough is optional
                    print(
                        f"[player] upnp start_passthrough failed "
                        f"(cast adopt): {exc!r}",
                        flush=True,
                    )
            self._emit()
            return self.snapshot()
        # Capture pieces we need to clean up OUTSIDE the lock so we
        # don't hold it during slow I/O (thread.join, decoder.close).
        old_thread = self._decoder_thread
        old_stop_flag = self._stop_flag
        old_decoder = self._decoder
        rate_matches = (
            pre.sample_rate == self._stream_sample_rate
            and pre.sd_dtype == self._stream_sd_dtype
            and pre.channels == self._stream_channels
        )
        old_stream = None if rate_matches else self._stream

        # Adopt the preload's pipeline. Callback is free to read
        # from the new queue the instant after these assignments.
        # `_swap_pipeline_to` writes rate/dtype/channels
        # unconditionally; that's a no-op in the same-rate case
        # because the values already match the active stream.
        self._swap_pipeline_to(pre)
        # Start DLNA passthrough for the new track.  This runs on
        # the HTTP thread, so blocking on SetAVTransportURI is fine.
        if (
            _upnp_manager is not None
            and _upnp_manager.is_active()
            and pre.source_urls is not None
        ):
            try:
                _upnp_manager.start_passthrough(
                    pre.source_urls,
                    prefetched=None,
                    metadata=getattr(pre, "track_meta", None) or self._current_track_meta,
                    track_id=getattr(pre, "track_id", None),
                )
            except Exception as exc:  # noqa: BLE001 - passthrough is optional
                print(
                    f"[player] upnp start_passthrough failed "
                    f"(adopt_preload): {exc!r}",
                    flush=True,
                )
        self._preload = None
        self._state = "playing"
        self._last_error = None
        self._seq += 1

        # The old pipeline is detached; release it without holding the
        # lock for its thread to unwind.
        self._retire_pipeline(old_stop_flag, old_decoder, old_thread)
        if old_stream is not None:
            # abort() over stop() — same reason as _teardown above:
            # stop() drains the buffer and stalls the cross-rate
            # bridge by half a second. Any tail audio we discard
            # is about to be drowned by the new stream anyway.
            try:
                old_stream.abort()
            except Exception:
                pass
            _close_pa_stream(old_stream)
            # Re-open at the new rate.
            self._open_output_stream(
                pre.sample_rate, pre.channels, pre.sd_dtype
            )
            try:
                self._start_stream(self._stream)
            except Exception as exc:
                log.exception("adopt_preload: stream.start failed")
                self._last_error = str(exc)
                self._state = "error"
                self._seq += 1

        self._emit()
        return self.snapshot()

    @staticmethod
    def _pull_block(q, carry, done_event, n: int, channels: int):
        """Pull up to `n` frames from (queue, carry) into a fresh
        (n, channels) float32 buffer for crossfade mixing.

        Returns (buf, new_carry, frames_filled, source_done). Unfilled
        frames are left as zeros — either a transient underrun or the
        source ending. `source_done` is True ONLY when the queue is empty
        AND `done_event` is set (a true end, not a hiccup), which is how
        the caller distinguishes "fade finished because the outgoing track
        ran out" from "decoder briefly behind".
        """
        buf = np.zeros((n, channels), dtype=np.float32)
        filled = 0
        source_done = False
        while filled < n:
            if carry is None:
                try:
                    chunk = q.get_nowait()
                except queue.Empty:
                    chunk = None
                if chunk is None:
                    source_done = done_event.is_set()
                    break
                carry = chunk
            take = min(n - filled, carry.shape[0])
            src = carry[:take]
            # Normalize integer PCM to float32 [-1, 1] so the equal-power
            # mix math works regardless of stream dtype. float32 chunks pass
            # through as-is (already in range).
            if src.dtype == np.int32:
                src = src.astype(np.float32) / 2147483648.0
            elif src.dtype == np.int16:
                src = src.astype(np.float32) / 32768.0
            if src.shape[1] == channels:
                buf[filled : filled + take] = src
            elif src.shape[1] == 2 and channels == 1:
                buf[filled : filled + take, 0] = src.mean(axis=1)
            # else: channel mismatch can't reach here (the crossfade is
            # gated on the incoming channel count matching the stream);
            # leave zeros rather than risk a malformed write.
            filled += take
            if take >= carry.shape[0]:
                carry = None
            else:
                carry = carry[take:]
        return buf, carry, filled, source_done

    def _maybe_begin_crossfade(self) -> None:
        """Start a crossfade if the current track is within the configured
        fade window and a compatible, sufficiently-buffered preload is
        ready. Realtime-safe: claims the preload under a NON-blocking lock
        (same pattern as `_try_gapless_swap`) and bails on any contention
        or unmet precondition, leaving normal playback untouched. Only ever
        called from the audio callback."""
        cf = self._crossfade_s
        if cf <= 0.0 or self._exclusive_mode:
            return
        rate = self._stream_sample_rate or 0
        if rate <= 0 or self._current_duration_ms <= 0:
            return
        total = int(cf * rate)
        if total <= 0:
            return
        track_samples = int(self._current_duration_ms / 1000.0 * rate)
        if track_samples - self._samples_emitted > total:
            return  # not in the fade window yet
        if self._preload is None:
            return
        if not self._lock.acquire(blocking=False):
            return  # contended — try again next callback
        try:
            pre = self._preload
            if pre is None:
                return
            if getattr(pre, "cast", False) is True:
                return  # cast records carry no PCM to crossfade
            if (
                pre.sample_rate != self._stream_sample_rate
                or pre.sd_dtype != self._stream_sd_dtype
                or pre.channels != self._stream_channels
            ):
                return  # incompatible — let the gapless/bridge path handle it
            if (
                pre.queue.qsize() < _XFADE_MIN_PREBUFFER_CHUNKS
                and not pre.done.is_set()
            ):
                return  # not buffered enough yet; re-check next callback
            # Claim it: _drop_preload / _try_gapless_swap won't touch it now.
            self._preload = None
        finally:
            self._lock.release()
        self._xfade_in = pre
        self._xfade_carry_in = None
        self._xfade_pos = 0
        self._xfade_total = total
        self._xfade_active = True
        print(
            f"[crossfade] begin -> incoming={pre.track_id} over {cf:.1f}s ({total} samples)",
            flush=True,
        )

    def _crossfade_fill(self, outdata, frames: int, channels: int) -> None:
        """Fill `outdata` with one callback's equal-power mix of the
        outgoing (current) and incoming (claimed preload) tracks, advance
        the fade, and promote the incoming track to primary once the fade
        completes or the outgoing track ends. Works for any stream dtype
        (float32, int32, int16): _pull_block normalizes chunks to float32
        for mixing, then the result is scaled back to the stream dtype
        before writing to outdata. Does NOT touch `_samples_emitted` /
        `_seq` — the shared post-fill below the call site handles those."""
        pre = self._xfade_in
        if pre is None:
            # Defensive — never mix without an incoming source.
            self._xfade_active = False
            outdata.fill(0)
            return
        out_buf, self._callback_carry, _of, out_done = self._pull_block(
            self._pcm_queue, self._callback_carry, self._decoder_done,
            frames, channels,
        )
        in_buf, self._xfade_carry_in, _if, _in_done = self._pull_block(
            pre.queue, self._xfade_carry_in, pre.done, frames, channels,
        )
        mixed = mix_crossfade_block(out_buf, in_buf, self._xfade_pos, self._xfade_total)
        # _pull_block normalizes all dtypes to float32 [-1, 1]; scale back
        # to the stream dtype before writing. float32 streams get a direct
        # copy; integer streams are scaled and clipped to their full range.
        if outdata.dtype == np.float32:
            outdata[:] = mixed
        elif outdata.dtype == np.int32:
            outdata[:] = np.clip(
                mixed * 2147483648.0, -2147483648.0, 2147483647.0
            ).astype(np.int32)
        elif outdata.dtype == np.int16:
            outdata[:] = np.clip(
                mixed * 32768.0, -32768.0, 32767.0
            ).astype(np.int16)
        self._xfade_pos += frames
        # Fade complete, or the outgoing track ran out before it finished
        # (the window started a hair late, or a short track) — promote the
        # incoming track either way.
        if self._xfade_pos >= self._xfade_total or out_done:
            self._promote_crossfade_incoming()

    def _promote_crossfade_incoming(self) -> None:
        """Make the faded-in track the primary pipeline. Mirrors
        `_try_gapless_swap`'s ended -> swap -> playing emit so the
        frontend's advance logic wires up the new track, then restores the
        incoming track's already-elapsed position + carry (which
        `_swap_pipeline_to` zeroes). Runs on the realtime callback thread,
        so it never JOINs the outgoing decoder — it signals it and lets the
        daemon thread exit (the same deadlock avoidance the file relies on
        everywhere)."""
        pre = self._xfade_in
        if pre is None:
            self._xfade_active = False
            return
        elapsed = self._xfade_pos
        carry_in = self._xfade_carry_in
        old_decoder = self._decoder
        old_stop = self._stop_flag
        print(f"[crossfade] promote -> {pre.track_id}", flush=True)
        self._state = "ended"
        self._seq += 1
        self._emit()
        self._swap_pipeline_to(pre)
        # The incoming track already played `elapsed` samples during the
        # fade; _swap_pipeline_to reset these to 0 / None.
        self._samples_emitted = elapsed
        self._callback_carry = carry_in
        self._state = "playing"
        self._seq += 1
        self._emit()
        self._xfade_active = False
        self._xfade_in = None
        self._xfade_carry_in = None
        self._xfade_pos = 0
        self._xfade_total = 0
        # Tear down the outgoing decoder WITHOUT joining (realtime thread).
        if old_stop is not None:
            old_stop.set()
        if old_decoder is not None:
            try:
                old_decoder.cancel_source()
            except Exception:
                pass

    def _abort_crossfade(self) -> None:
        """Cancel an in-progress (or claimed-but-not-yet-started)
        crossfade and dispose the incoming preload we'd claimed. Called
        from the pipeline-reset paths (teardown / seek). A manual skip,
        stop, or seek inside the fade window would otherwise leak the
        claimed decoder and leave stale fade state. Safe to call when no
        crossfade is active.

        Disposal is by `stop_flag` + `cancel_source` only — both are
        designed to be called concurrently with the decoder thread (the
        teardown path does the same on the primary decoder). We do NOT
        synchronously `close()` the decoder here: a callback could still
        be reading the claimed preload's already-decoded queue, and the
        daemon decoder thread exits on the stop flag and GCs on its own."""
        pre = self._xfade_in
        self._xfade_active = False
        self._xfade_in = None
        self._xfade_carry_in = None
        self._xfade_pos = 0
        self._xfade_total = 0
        if pre is not None:
            try:
                pre.stop_flag.set()
            except Exception:
                pass
            try:
                pre.decoder.cancel_source()
            except Exception:
                pass

    def _renderer_clock_active(self) -> bool:
        """True when a bounded per-track DLNA file is being served AND
        the renderer has confirmed it reports its own playback position.

        In that mode the renderer — not the muted desktop decoder — is
        the playback clock, so the player must not advance the queue on
        its own (earlier) EOF. Lock-free (one attribute read on the
        manager) so the realtime audio callback can gate on it.
        Chromecast never serves a bounded UpnpManager file, so this is
        always False for Cast and for local playback."""
        return bool(
            _upnp_manager is not None
            and _upnp_manager.is_active()
            and _upnp_manager.renderer_clock_active()
        )

    def _upnp_consumed_track_id(self) -> Optional[str]:
        """The TIDAL track id the renderer auto-advanced into, if the
        UpnpManager recorded it. Sticky: it survives the watchdog's
        generation reset and is cleared only when a new next is staged,
        so the evidence is still available after the frontend's
        follow-up play_track(next). None when there is no manager or no
        consumption was recorded."""
        mgr = _upnp_manager
        if mgr is None or not mgr.is_active():
            return None
        getter = getattr(mgr, "last_consumed_track_id", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def on_renderer_track_ended(self) -> None:
        """Called by the UpnpManager watchdog when the renderer reaches
        the end of the current bounded DLNA track.

        Emits `ended` so the frontend advances its queue (it owns the
        queue). When the manager recorded that the renderer consumed the
        pre-staged next URI (positive evidence), the matching preload is
        adopted and `playing` is emitted for the new track, so the
        frontend's redundant play_track(next) lands on Path 0. Without
        evidence nothing is adopted — the announce path advances instead
        — because adopting would show N+1 in the UI while the renderer is
        still on N. Runs on the watchdog thread, not the realtime
        callback, so emitting snapshots is safe. No-op unless a bounded
        DLNA renderer clock is actually driving playback."""
        if not self._renderer_clock_active():
            return
        consumed = self._upnp_consumed_track_id()
        with self._lock:
            if self._state not in ("playing", "paused"):
                return
            # Idempotency: a duplicate fire for an advance already
            # adopted must not emit a second `ended`.
            if consumed is not None and self._current_track_id == consumed:
                return
            # ended -> frontend advance; playing -> so play_track(next)
            # takes the adopt path rather than a full teardown.
            self._state = "ended"
            self._seq += 1
            self._emit()
            pre = self._preload
            if pre is not None and getattr(pre, "cast", False) is True:
                if consumed is not None and pre.track_id == consumed:
                    # Renderer-clock advance is backend-driven: adopt the
                    # pre-staged next track here so playback and the
                    # backend's current-track state move even if the
                    # frontend misses or drops the single `ended` frame.
                    # The following `playing` snapshot carries the new
                    # track id, which the frontend uses to re-sync and
                    # preload the track after it.
                    self._adopt_preload_locked(pre)
                    return
                # Cast preload but no positive evidence: keep it for the
                # frontend's play_track(next) to adopt; the announce path
                # handles the actual switch.
                self._state = "playing"
                self._seq += 1
                return
            if pre is not None:
                self._adopt_preload_locked(pre)
                return
            self._state = "playing"
            self._seq += 1

    def _try_gapless_swap(self) -> bool:
        """Called from the audio callback when the current queue is
        empty + primary decoder is done. Swaps in the preloaded
        track's decoder/queue/thread if the sample rate + dtype +
        channel count match. Returns True on success; False if
        there's no preload or it's incompatible.

        Emits `ended` BEFORE the swap and `playing` after. The
        frontend's advance logic fires on `ended`, updates its
        `expectedTrackIdRef` via `playAtIndex`, and the redundant
        `play_track(next)` that follows lands on Path 0 (no-op
        because current_track_id already matches). Without the
        `ended` emit, the frontend's expected-guard would drop all
        new snapshots because its expected is still the previous
        track.

        Assignments are individually atomic under the GIL; the
        callback is the sole modifier during the track-boundary
        moment.
        """
        # Renderer-clock mode: the renderer is the playback clock, so the
        # muted desktop decoder's EOF must NOT advance the queue — the
        # UpnpManager watchdog emits `ended` at the renderer's real end.
        # Bail to CallbackStop -> _on_stream_finished, which keeps the
        # preload alive for the renderer-driven adopt.
        if self._renderer_clock_active():
            print(
                "[pcm] _try_gapless_swap bailed: renderer clock active",
                flush=True,
            )
            return False
        # Atomically capture + clear the preload slot so a racing
        # `_drop_preload` on the HTTP thread can't close the
        # preload's decoder between our capability check below and
        # the ref swap. The lock is held only for the pointer grab
        # (microseconds) — not during the actual swap or emit.
        #
        # CRITICAL: this runs on the PortAudio realtime callback
        # thread, and _lock is also held by the load/adopt path
        # while it join()s decoder threads that can stall on a
        # network segment read. A *blocking* acquire here parks the
        # realtime thread on that lock until the join unwinds —
        # which, if the decoder is wedged on a hung fetch, is
        # forever: audio dies with no log line and every thread ends
        # up in a cond-wait (the silent freeze). So acquire
        # non-blocking only; on contention, bail and let the
        # off-realtime _on_stream_finished bridge handle the
        # transition (a momentary gap instead of a permanent
        # deadlock). This restores the "the audio callback is
        # lock-free" invariant the rest of this file relies on.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            pre = self._preload
            if pre is None:
                return False
            if (
                pre.sample_rate != self._stream_sample_rate
                or pre.sd_dtype != self._stream_sd_dtype
                or pre.channels != self._stream_channels
            ):
                # Incompatible — fall through to CallbackStop and
                # let _on_stream_finished spawn the bridge thread.
                return False
            self._preload = None
        finally:
            self._lock.release()
        log.info(
            "gapless swap (inline) -> track=%s rate=%dHz dtype=%s",
            pre.track_id, pre.sample_rate, pre.sd_dtype,
        )
        # Phase 1: emit `ended` so the frontend's advance logic
        # wires up expectedTrackIdRef for the new track.
        self._state = "ended"
        self._seq += 1
        self._emit()
        # Phase 2: actual swap. Old decoder + thread have already
        # finished (done event is set), we just replace refs.
        self._swap_pipeline_to(pre)
        # Defer UPnP passthrough notification: the audio callback
        # can't block on HTTP (SetAVTransportURI).  The flag is
        # picked up by _load_locked() Path 0 when the frontend's
        # play_track() arrives.
        if (
            _upnp_manager is not None
            and _upnp_manager.is_active()
            and pre.source_urls is not None
        ):
            self._pending_upnp_notify = True
        self._state = "playing"
        self._seq += 1
        self._emit()
        return True

    def _on_stream_finished(self) -> None:
        # Called by sounddevice when the stream ends. Three reasons it
        # might fire:
        #   1. Natural EOF — callback raised CallbackStop because the
        #      decoder finished and the queue drained.
        #   2. Intentional stream replacement (set_output_device,
        #      set_exclusive_mode, cross-rate bridge) — `_replacing_
        #      stream` is set; ignore this firing entirely, the
        #      replacement code is opening a new stream right now.
        #   3. Device loss — headphones unplugged, USB DAC pulled,
        #      Bluetooth out of range. PortAudio aborts the stream
        #      because the underlying CoreAudio device is gone, but
        #      our decoder still has frames buffered. Recover by
        #      reopening on the system default so playback continues
        #      on speakers without pausing — that's how every other
        #      streaming app behaves and what users expect.
        #
        # Capture AND clear the preload pointer under the same lock
        # acquisition so a concurrent _drop_preload on the HTTP
        # thread (e.g. if the user also clicked Stop at the same
        # microsecond as natural EOF) cannot close the preload's
        # decoder between our read of self._preload and the bridge
        # thread's use of it. We "own" the preload for the duration
        # of the bridge attempt; if the bridge fails, it's
        # responsible for disposing of the decoder itself.
        with self._lock:
            if self._state in ("idle", "error"):
                return
            if self._replacing_stream:
                # Intentional stream swap — see set_output_device /
                # set_exclusive_mode. Don't transition state; don't
                # trigger device-loss recovery.
                return
            # Device-loss heuristic: stream ended unexpectedly, with
            # work still pending. Two ways to know the work isn't done:
            #
            #   1. Decoder hasn't finished decoding the rest of the
            #      track (`not decoder_done`).
            #   2. Decoder IS done but the PCM queue still has chunks
            #      the callback never got to render (`qsize > 0`). This
            #      is the case when the device disappears late in a
            #      short track — the decoder finished filling the queue
            #      before the unplug, so `decoder_done` is True even
            #      though several seconds of audio are still waiting to
            #      play.
            #
            # Natural EOF, by contrast, has both the decoder done AND
            # the queue drained — the callback got everything out
            # before raising CallbackStop.
            #
            # `paused` counts as active here: the callback never drains
            # while paused, so a friendly end is impossible, and
            # Windows reports a Spatial Sound change (enable "Dolby
            # Atmos for Headphones") by killing the stream exactly this
            # way. A preload no longer exempts the stream either —
            # bridging to the next track on a dead endpoint just moved
            # the failure one track later. Never emit `ended` for this:
            # the frontend treats `ended` as "advance" and the whole
            # queue skips itself.
            decoder_done = self._decoder_done.is_set()
            queue_pending = (
                self._pcm_queue.qsize() > 0
                if self._pcm_queue is not None
                else False
            )
            was_active = self._state in ("playing", "paused")
            pre = self._preload
            should_recover = was_active and (
                not decoder_done or queue_pending
            )
            if should_recover:
                log.info(
                    "device-loss detected: triggering recovery "
                    "(decoder_done=%s, queue_pending=%s)",
                    decoder_done, queue_pending,
                )
                # Recovery runs off-thread because the finished_callback
                # runs on sounddevice's own thread which we shouldn't
                # re-enter stream machinery from.
                threading.Thread(
                    target=self._recover_from_device_loss,
                    name="pcm-device-recovery",
                    daemon=True,
                ).start()
                return
            # Renderer-clock mode (or a metadata-only cast preload): the
            # muted local stream ended early by design; the renderer's
            # watchdog will emit `ended` at the real end. Close the dead
            # local stream but keep the preload and the 'playing' state so
            # the pending play_track(next) adopts the preload. A cast
            # preload has NO local pipeline (no decoder, no dtype), so it
            # must never fall through to the cross-rate bridge — that
            # re-opened an OutputStream with dtype '' and raised
            # "Invalid output sample format". Only when a real preload
            # actually exists — with none there is nothing for the
            # renderer-driven adopt to take, so fall through to the normal
            # `ended` instead of waiting on a watchdog that has nothing to
            # advance into.
            is_cast = pre is not None and getattr(pre, "cast", False) is True
            renderer_clock = pre is not None and (
                is_cast or self._renderer_clock_active()
            )
            if renderer_clock:
                bound_stream = self._stream
                self._stream = None
                print(
                    "[pcm] EOF: renderer clock active, keeping preload "
                    f"preload={getattr(self._preload, 'track_id', None)}",
                    flush=True,
                )
            else:
                bound_stream = None
                self._preload = None

        if renderer_clock:
            _close_pa_stream(bound_stream)
            return

        # Cross-rate bridge: preload exists but has a different
        # sample rate / dtype than the current stream, so the
        # callback-level gapless swap was skipped. Re-open the
        # OutputStream with the preload's params. Off-thread
        # because sounddevice's finished_callback runs on its
        # own thread which shouldn't re-enter stream machinery.
        if pre is not None:
            threading.Thread(
                target=self._bridge_to_preload,
                args=(pre,),
                name="pcm-bridge",
                daemon=True,
            ).start()
            return

        # Natural EOF. PortAudio's `paComplete` puts the stream
        # into a terminal "stopped" state that `Pa_StartStream`
        # cannot revive — the docs say it has to be reopened. If
        # we leave `self._stream` pointing at this corpse, the
        # next `load()` snapshots it as `active_stream`, takes
        # path (A) ("kept stream open"), and rebinds the new
        # pipeline against a dead OutputStream. `play()` calls
        # `stream.start()` on macOS, which returns without error
        # but never re-wakes the CoreAudio callback thread. The
        # decoder fills the queue, blocks on `put`, and the
        # stall watchdog reports `state=playing, silent`. Close
        # and drop the ref here so the next load takes the full
        # reopen path (B) — that path already works correctly.
        old_stream = self._stream
        with self._lock:
            self._stream = None
            self._transition("ended")
        _close_pa_stream(old_stream)
        # Signal source_done to the DLNA passthrough encoder so the
        # renderer sees a clean end-of-stream instead of hanging on
        # the HTTP connection until the 30s write timeout kicks in.
        # Only when no preload exists — if one is pending, the gapless
        # transition will start passthrough for the next track.
        if _upnp_manager is not None:
            try:
                if _upnp_manager.is_active():
                    _upnp_manager.signal_source_done()
            except Exception as exc:
                print(
                    f"[player] upnp signal_source_done failed: {exc!r}",
                    flush=True,
                )
        self._emit()

    def _recover_from_device_loss(self) -> None:
        """Re-bind the output stream to a working device after a
        CoreAudio routing change. Two scenarios this covers:

          A. The currently-selected device disappeared (headphones
             unplugged, USB DAC removed, AirPods drifted out of
             range). resolve_output_device reports it gone, so we
             drop the selection and reopen on the system default.

          B. The device list changed but the selected device is
             still around (a NEW device just plugged in — e.g.
             headphones reconnected while audio is on speakers).
             The reopen on the original device succeeds; the user's
             selection is preserved AND PortAudio's enumeration
             gets refreshed in the same step so the picker shows
             the new device next time it opens.

        Standard streaming-app behavior: unplug headphones, audio
        keeps playing on speakers; plug them back in, the picker
        sees them again. ~50 ms of silence while CoreAudio
        renegotiates, same as a manual device switch.

        The decoder thread keeps feeding the PCM queue throughout,
        so once we open a fresh stream the first callback drains
        the queued frames and audio resumes seamlessly.
        """
        with self._pipeline_lock:
            with self._lock:
                # Bail if state moved on — user pressed stop, a
                # natural EOF arrived, error path fired, etc.
                if self._state in ("idle", "error", "ended"):
                    return
                if self._stream is None:
                    return
                sample_rate = self._stream_sample_rate
                channels = self._stream_channels
                sd_dtype = self._stream_sd_dtype
                old_stream = self._stream
                self._stream = None
                self._callback_carry = None
                # KEEP the user's selection. We try it first below;
                # only fall back to default if the reopen fails
                # (which is what means "the device actually
                # disappeared" vs "device list just changed").
                original_device_id = self._selected_device_id
                self._replacing_stream = True

            if (
                sample_rate is None
                or channels is None
                or sd_dtype is None
            ):
                with self._lock:
                    self._replacing_stream = False
                return

            _close_pa_stream(old_stream)

            # Force PortAudio to re-enumerate so devices that
            # disappeared/plugged in since the last init are seen.
            # pa_guard no-ops if another stream is still open.
            self._reinit_portaudio()

            # _open_output_stream now falls back to the system default
            # on its own when the saved device is gone (it no longer
            # raises), so scenario A can't be inferred from an exception
            # any more. Ask the resolver directly: if the selection
            # really vanished, drop it up front so the picker and status
            # stop reporting a device that isn't there, and the reopen
            # below lands on the default. A present-but-unopenable device
            # still falls through to the second attempt via the except.
            _, device_available = resolve_output_device(original_device_id)
            device_gone = bool(original_device_id) and not device_available
            if device_gone:
                with self._lock:
                    self._selected_device_id = ""

            # First attempt: the resolved device (scenario B), or the
            # system default when the pick vanished (scenario A).
            try:
                with self._lock:
                    self._open_output_stream(
                        sample_rate, channels, sd_dtype
                    )
                self._start_stream(self._stream)
                if device_gone:
                    log.info(
                        "device-recovery: original device id=%r gone; "
                        "reopened on system default (rate=%d, channels=%d)",
                        original_device_id, sample_rate, channels,
                    )
                else:
                    log.info(
                        "device-recovery: reopened on device id=%r "
                        "(rate=%d, channels=%d)",
                        self._selected_device_id or "system default",
                        sample_rate, channels,
                    )
                with self._lock:
                    self._replacing_stream = False
                return
            except Exception as exc:
                log.info(
                    "device-recovery: original device id=%r "
                    "unavailable (%r); trying another host-API copy",
                    original_device_id, exc,
                )
                # The name still resolved, so the endpoint exists and
                # only this host API's client is unusable — the state a
                # Windows Spatial Sound change leaves the WASAPI copy
                # in. MME/DirectSound are separate clients on the same
                # speakers and keep working.
                try:
                    with self._lock:
                        self._open_stream_tier("alternate-hostapi")
                    log.info(
                        "device-recovery: reopened on another host-API "
                        "copy of device id=%r",
                        original_device_id,
                    )
                    with self._lock:
                        self._replacing_stream = False
                    return
                except Exception as alt_exc:
                    log.info(
                        "device-recovery: alternate host-API copy of "
                        "id=%r also failed (%r)",
                        original_device_id, alt_exc,
                    )
                # Reset state for the default-device attempt.
                with self._lock:
                    self._selected_device_id = ""
                    self._stream = None
                    self._callback_carry = None

            # Second attempt: system default (scenario A: device
            # actually disappeared).
            try:
                with self._lock:
                    self._open_output_stream(
                        sample_rate, channels, sd_dtype
                    )
                self._start_stream(self._stream)
                log.info(
                    "device-recovery: reopened on system default "
                    "(rate=%d, channels=%d)",
                    sample_rate, channels,
                )
            except Exception:
                log.exception(
                    "device-recovery: system-default fallback also failed"
                )
                with self._lock:
                    self._transition("error")
                    self._stream = None
                self._emit()
            finally:
                with self._lock:
                    self._replacing_stream = False

    def _bridge_to_preload(self, pre: _Preload) -> None:
        """Stream re-open + swap for a cross-rate preload. Produces
        ~50ms of silence between tracks — same limitation every
        gapless player has when sample rates change.

        Serialized on `_pipeline_lock` so a concurrent `stop()` /
        `load()` on the HTTP thread can't run teardown while we're
        mid-bridge. If teardown drove us to idle before we acquired
        the lock, the preload we captured may have been dropped —
        bail out and let the user-initiated state stand.
        """
        with self._pipeline_lock:
            with self._lock:
                if self._state == "idle":
                    # Teardown ran between _on_stream_finished and
                    # here. Our captured `pre` may have been closed
                    # by _drop_preload; don't try to adopt it.
                    try:
                        pre.stop_flag.set()
                        pre.decoder.close()
                    except Exception:
                        pass
                    return

            # Mirror `_try_gapless_swap`'s pattern: emit `ended` for
            # the OUTGOING track BEFORE swapping. The frontend's
            # advance logic keys off this snapshot to wire its
            # `expectedTrackIdRef` up to the new track id; without
            # the emit, every subsequent snapshot for the new track
            # fails the late-echo guard and the now-playing bar gets
            # stuck on the previous track even though audio has
            # moved on. Same-rate transitions don't hit this path
            # (they go through `_try_gapless_swap` which already
            # emits `ended`), so the bug only surfaced on cross-rate
            # transitions — typically album → Artist Radio where
            # the radio tracks have a different sample rate than
            # the album that just finished.
            with self._lock:
                self._state = "ended"
                self._seq += 1
            self._emit()

            log.info(
                "gapless bridge (cross-rate) -> track=%s rate %d->%dHz dtype %s->%s",
                pre.track_id,
                self._stream_sample_rate or 0,
                pre.sample_rate,
                self._stream_sd_dtype,
                pre.sd_dtype,
            )
            # The old stream already fired finished_callback, so it's
            # stopped. Close it for good measure and release
            # resources.
            old_stream = self._stream
            self._stream = None
            _close_pa_stream(old_stream)
            # Close the old decoder (thread has already exited since
            # decoder_done fired).
            old_decoder = self._decoder
            if old_decoder is not None:
                try:
                    old_decoder.close()
                except Exception:
                    pass

            with self._lock:
                # Clear the preload slot BEFORE opening a new
                # stream so a racing preload() call (unlikely this
                # early) can't re-stomp it.
                self._preload = None
                self._swap_pipeline_to(pre)
                # Start DLNA passthrough for the new track.  The
                # bridge thread is off-realtime, so HTTP is fine.
                if (
                    _upnp_manager is not None
                    and _upnp_manager.is_active()
                    and pre.source_urls is not None
                ):
                    try:
                        _upnp_manager.start_passthrough(
                            pre.source_urls,
                            prefetched=None,
                            metadata=getattr(pre, "track_meta", None) or self._current_track_meta,
                            track_id=getattr(pre, "track_id", None),
                        )
                    except Exception as exc:  # noqa: BLE001 - passthrough is optional
                        print(
                            f"[player] upnp start_passthrough failed "
                            f"(bridge): {exc!r}",
                            flush=True,
                        )
                self._open_output_stream(
                    pre.sample_rate, pre.channels, pre.sd_dtype
                )
                self._transition("playing")
            try:
                self._start_stream(self._stream)
            except Exception:
                log.exception("bridge: stream.start failed")
                with self._lock:
                    self._last_error = "stream reopen failed"
                    self._transition("error")
                self._emit()
                return
            self._emit()

    def _drain_queue_locked(self) -> None:
        while not self._pcm_queue.empty():
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                break

    def _transition(self, state: str, track_id: Optional[str] = None) -> None:
        self._state = state
        if track_id is not None:
            self._current_track_id = track_id
        self._seq += 1

    def _emit(self) -> None:
        snap = self.snapshot()
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(snap)
            except Exception:
                log.exception("listener raised")

    def _discard_pipeline(self, pre: "_Preload") -> None:
        """Stop a decoder thread whose track will never be adopted."""
        pre.stop_flag.set()
        decoder = pre.decoder
        if decoder is None:
            return
        try:
            decoder.cancel_source()
        except Exception:
            pass
        if pre.thread is not None and pre.thread.is_alive():
            pre.thread.join(timeout=0.5)
        try:
            decoder.close()
        except Exception:
            pass

    def _swap_pipeline_to(self, pre: "_Preload") -> None:
        """Replace the active pipeline refs with those from `pre`.

        Single source of truth for the field copies that drive a
        track swap. Same shape was hand-rolled in three places
        (_try_gapless_swap, _bridge_to_preload, _adopt_preload_locked)
        and inevitably drifted — the cross-rate gapless bug we just
        shipped a fix for came from one of those copies forgetting a
        partner emit. Centralising it here means a new field on
        `_Preload` only has to be wired up once.

        Atomicity: each assignment is one bytecode op under the GIL,
        but the *sequence* is not atomic, and the audio callback is
        lock-free — holding `_lock` (as `_adopt_preload_locked` does)
        does NOT keep the callback out, so it could read a
        half-swapped pipeline: the new track's queue with the old
        track's ReplayGain and filter state. That mismatch is the
        split-second "explosion" on track change. The `_swapping`
        guard below closes it: it's set before the first ref changes
        and cleared after the ReplayGain re-derive, and the callback
        emits silence the whole time (sub-millisecond, inaudible).
        `_try_gapless_swap` runs in the callback thread itself, so
        the guard there is just set and cleared within one
        invocation — harmless, and keeps a single code path.

        Always copies the rate / dtype / channels triple along with
        the rest. Same-rate swaps no-op those (the values match the
        active stream); the alternative — conditional copies — was
        the contract drift this helper exists to prevent.
        """
        old_track = self._current_track_id
        old_info = self._current_stream_info
        self._swapping = True
        try:
            self._decoder = pre.decoder
            self._pcm_queue = pre.queue
            self._decoder_thread = pre.thread
            self._stop_flag = pre.stop_flag
            self._decoder_done = pre.done
            self._current_track_id = pre.track_id
            self._current_duration_ms = pre.duration_ms
            self._current_stream_info = pre.stream_info
            # Keep the now-playing metadata in sync with the adopted
            # track, and carry the preload's own meta so the swap
            # notification uses the track actually being adopted.
            if getattr(pre, "track_meta", None) is not None:
                self._current_track_meta = pre.track_meta
            self._source_urls = pre.source_urls
            self._source_path = pre.source_path
            self._stream_sample_rate = pre.sample_rate
            self._stream_sd_dtype = pre.sd_dtype
            self._stream_channels = pre.channels
            self._samples_emitted = 0
            self._callback_carry = None
            # Re-derive ReplayGain off the new track's tags while the
            # callback is still silenced, so it never applies the old
            # gain to the new samples.
            self._apply_replaygain_for(pre.stream_info)
        finally:
            self._swapping = False
        # Logged after the guard clears so the file write is outside
        # the silence window. This is the line that was missing when
        # the explosion couldn't be traced from logs: it records the
        # ReplayGain delta across the swap, which is what drives the
        # level jump when the race is open.
        def _rg(info) -> str:
            if info is None:
                return "n/a"
            t = getattr(info, "track_replay_gain_db", None)
            a = getattr(info, "album_replay_gain_db", None)
            return f"track={t} album={a}"

        try:
            audio_log.info(
                "swap %s -> %s | rg old(%s) new(%s)",
                old_track,
                pre.track_id,
                _rg(old_info),
                _rg(pre.stream_info),
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stream-info helpers (mutagen probe for local files)
# ---------------------------------------------------------------------------


def _safe_int(v: object) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _replaygain_tags_from(info: Optional[StreamInfo]) -> ReplayGainTags:
    """Lift the four ReplayGain numbers off a StreamInfo into the
    explicit tag dataclass the gain engine consumes. Returns an
    all-None tags object when info is None (idle player) so callers
    can apply unconditionally without an extra null check."""
    if info is None:
        return ReplayGainTags()
    return ReplayGainTags(
        track_gain_db=info.track_replay_gain_db,
        track_peak=info.track_peak,
        album_gain_db=info.album_replay_gain_db,
        album_peak=info.album_peak,
    )


def _tag_reference_shift_db(info: Optional[StreamInfo]) -> float:
    """Offset that moves tag-driven gain onto our -16 LUFS reference.

    TIDAL's own ReplayGain tags are normalized to about -14 LUFS, so
    they attenuate ~2 dB less than the same track does when we measure
    it locally (or than Apple's Sound Check does). Only stream tags
    need the shift: a local file's tags come from the user's tagger,
    which follows the ReplayGain spec instead.
    """
    if info is None or info.source == "local":
        return 0.0
    return loudness.TARGET_LUFS - TIDAL_TAG_REFERENCE_LUFS


def _safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    # NaN / inf would silently propagate into the gain math and either
    # mute audio (multiply by 0 from a NaN clamp) or blow it up
    # (multiply by inf). Treat them as missing so the caller sees None
    # and decides — the standard "default to 0 dB / unity" path.
    if out != out or out == float("inf") or out == float("-inf"):
        return None
    return out


def _normalize_codec(raw: object) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if "flac" in s:
        return "flac"
    if "alac" in s:
        return "alac"
    if "mp4a" in s or "aac" in s:
        return "aac"
    if "ec-3" in s or "eac3" in s:
        return "eac3"
    if "mp3" in s:
        return "mp3"
    if "opus" in s:
        return "opus"
    if "vorbis" in s:
        return "vorbis"
    return s


def _is_atmos_stream_info(info: Optional[StreamInfo]) -> bool:
    if info is None:
        return False
    return str(getattr(info, "audio_mode", "") or "").upper() == "DOLBY_ATMOS"


# Spatial formats that place Dolby Atmos objects. Any other active
# spatialiser (Windows Sonic, DTS, or none) virtualises the 7.1 bed the
# native pipeline hands it instead — the JOC objects are decoded but
# nothing puts them anywhere. Names match winrt_player.ps1's slugs.
_ATMOS_SPATIAL_FORMATS = frozenset(
    {"atmos-headphones", "atmos-home-theater", "atmos-speakers"}
)

# Codecs whose content is Dolby. Classification reads the codec rather
# than audio_mode: the local probe marks every >2-channel file as
# DOLBY_ATMOS (a 5.1 FLAC included), which would badge plain surround
# as Dolby.
_DOLBY_CODECS = frozenset({"eac3", "ac3", "ac4", "truehd"})


def _dolby_playback_kind(
    info: Optional[StreamInfo],
    *,
    native: bool,
    spatial_format: str,
    decoders: tuple[str, ...] | list[str] = (),
) -> Optional[str]:
    """How Dolby content is actually reaching the listener.

    "atmos"       the native pipeline decoded the JOC objects and a
                  Dolby Atmos spatial format on the endpoint placed them.
    "dolby_audio" the same mix is audible only as its 5.1/7.1 bed: the
                  native pipeline with a non-Atmos (or no) spatial
                  format, or any Dolby asset on the PCM path, where PyAV
                  drops the object metadata before Windows sees it.
    None          not Dolby content.

    `decoders` is the helper's loaded-module list. It is empty until
    MediaPlayer starts decoding, so an empty list means "not known
    yet", not "no Dolby decoder" — only a populated list without a
    Dolby module proves Windows fell back to a downmix.
    """
    if info is None:
        return None
    if str(info.codec or "").lower() not in _DOLBY_CODECS:
        return None
    if not native:
        return "dolby_audio"
    if str(spatial_format or "").lower() not in _ATMOS_SPATIAL_FORMATS:
        return "dolby_audio"
    if decoders and not any("dolby" in str(d).lower() for d in decoders):
        return "dolby_audio"
    return "atmos"


# Containers that can carry E-AC-3 JOC / AC-4. Checked before the
# libav probe so ordinary FLAC/MP3 local loads never pay for the
# extra container open the native-atmos decision needs.
_NATIVE_ATMOS_EXTS = frozenset({".m4a", ".mp4", ".mov", ".ac4"})

# A streamed Atmos track exists only as a DASH segment list, and the
# native pipeline opens one file. Segments are therefore concatenated
# into one file before playback. This bounds how long a load waits
# for that: past it the click answers with the PCM engine (the 5.1/7.1
# bed) rather than leaving the user in silence.
_NATIVE_STREAM_BUFFER_DEADLINE_S = 20.0
# How long a buffered track's in-memory record is reused without a
# re-read of its sidecar. The file itself is not bound by this — see
# the disk budget below.
_NATIVE_STREAM_TTL_S = 600.0
# A failed buffer retries sooner than a good one expires — a dropped
# connection is usually transient — but not on every click.
_NATIVE_STREAM_FAILURE_TTL_S = 60.0
# Where buffered tracks live. These were %TEMP%/tideway-atmos-* scratch
# dirs, which cost 20-30 MB per track, died with the process, and — the
# sweep nobody owned — were never deleted even then. Under the data dir
# a replay of the same Atmos track, including one after an app restart,
# opens the file instead of downloading it again.
_NATIVE_STREAM_CACHE_DIR = "atmos_cache"
# Each file is 20-30 MB, so this holds roughly an album and a half of
# Atmos material; past it the least recently downloaded files go first.
_NATIVE_STREAM_DISK_MAX_BYTES = 512 * 1024 * 1024
# Files nobody has touched in a month are not a replay worth keeping.
_NATIVE_STREAM_DISK_MAX_AGE_S = 30 * 24 * 3600
# DASH playback: how long the click waits for the init segment and the
# first media fragment. Measured at ~0.5 s on a cold track and near
# zero when the hover/album prefetch already seeded them; the bound only
# decides how quickly a stalled download hands the track to the PCM bed.
_DASH_PRIME_DEADLINE_S = 8.0
# A DASH write keeps going long after playback starts; this is its own
# bound, mirroring the preload's.
_DASH_DEADLINE_S = 180.0
# If the helper rejects the adaptive source, the download is still
# running: how long the fallback waits for the joined file.
_DASH_FALLBACK_WAIT_S = 60.0
# How often the click re-checks the writer while waiting to prime.
_DASH_POLL_S = 0.02
# After a give-up, how long to wait for the cancelled writer to close
# its handle before the whole-file fallback rewrites the same path.
_DASH_CANCEL_WAIT_S = 2.0
# How long a helper opened by the look-ahead stays worth adopting. Its
# session is closed when the queue moves on, so this only retires one
# nobody ever clicked.
_WARM_HELPER_TTL_S = 300.0
# Advisory bitrate in the manifest. One representation, so nothing
# selects on it; Media Foundation only wants a plausible number.
_DASH_BANDWIDTH_BPS = 800000
# Attempts to delete a retired segment directory. MediaPlayer releases
# the files a moment after `Source` is replaced, so the first tries can
# lose a sharing race; the startup sweep takes whatever is left.
_DASH_DIR_DELETE_TRIES = 5
_DASH_DIR_DELETE_DELAY_S = 0.4


@dataclass(eq=False)
class _ResolveTicket:
    """One in-flight manifest resolve, shared by every caller that
    wants the same (track, quality).

    `event` fires when the leader is done; `result` / `error` carry
    what it found. A waiter that wakes to neither (its entry was
    evicted from the cache) resolves on its own.
    """

    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[tuple] = None
    error: Optional[BaseException] = None


@dataclass
class _BufferedAtmos:
    """One streamed Atmos track materialized as a local file."""

    path: str
    duration_s: Optional[float]
    info: StreamInfo
    expires_at: float


@dataclass(eq=False)
class _DashAtmos:
    """A streamed Atmos track served to MediaPlayer as DASH.

    `primed` fires once the init segment and the first media segment
    exist, `done` when the download finished (or failed), and `cancel`
    is how a new load or a stop stops the download. `done_failed` is
    written before `done` so a waiter can tell a finish from a give-up.
    `cache_path` is the joined file, which only exists after `done`.

    One session serves a track whether a click started it or a
    look-ahead preload did; identity, not value, decides whether a
    session is the current one — two loads of the same track produce
    identical field values.
    """

    track_id: str
    cache_path: str
    directory: str
    url: str
    publication: DashPublication
    duration_s: Optional[float]
    info: StreamInfo
    primed: threading.Event
    done: threading.Event
    cancel: threading.Event
    done_failed: bool = False


def _remove_dash_dir(directory: str) -> None:
    """Delete a retired segment directory, retrying past the point
    where MediaPlayer still holds the files open."""
    for attempt in range(_DASH_DIR_DELETE_TRIES):
        if attempt:
            time.sleep(_DASH_DIR_DELETE_DELAY_S)
        try:
            shutil.rmtree(directory)
            return
        except FileNotFoundError:
            return
        except OSError:
            continue
    log.debug("atmos dash: leaving %s for the startup sweep", directory)


def _atmos_sidecar_path(path: str) -> str:
    return path + ".json"


def _write_atmos_sidecar(
    path: str, duration_s: Optional[float], info: StreamInfo
) -> None:
    """Persist what the buffered file is, next to the file itself.

    The player needs more than the container says: `source` and
    `audio_quality` select the now-playing badge, and the ReplayGain
    tags are TIDAL's own normalization data, which no probe can
    recover from the bytes. Without this sidecar a re-adopted file fell
    back to an R128 scan of every Atmos track after each restart.

    The size is recorded so a later read can tell a finished file from
    one truncated by a crash — `os.replace` publishes the audio and the
    sidecar separately, so both need a way to agree they belong
    together.
    """
    sidecar = _atmos_sidecar_path(path)
    partial = sidecar + ".part"
    try:
        payload = {
            "version": 1,
            "size": os.path.getsize(path),
            "duration_s": duration_s,
            "info": asdict(info),
        }
        with open(partial, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(partial, sidecar)
    except OSError as exc:
        log.debug("atmos cache sidecar write failed for %s: %s", path, exc)
        try:
            os.remove(partial)
        except OSError:
            pass


def _read_atmos_sidecar(path: str) -> Optional[dict]:
    """The sidecar's payload, or None when it is missing/unreadable or
    describes a different file than the one on disk."""
    try:
        with open(_atmos_sidecar_path(path), encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return None
        if payload.get("size") != os.path.getsize(path):
            return None
        if not isinstance(payload.get("info"), dict):
            return None
    except (OSError, ValueError):
        return None
    return payload


def _remove_atmos_cache_file(path: str) -> None:
    """Drop a buffered file and its sidecar, best effort."""
    for target in (path, _atmos_sidecar_path(path)):
        try:
            os.remove(target)
        except OSError:
            pass


def _sweep_legacy_atmos_dirs() -> None:
    """Delete `tideway-atmos-*` scratch dirs in the temp folder.

    Builds before the persistent cache buffered each track into its own
    mkdtemp() directory and never removed it: 20-30 MB per track, and
    hundreds of MB had accumulated. The prefix is specific enough to
    own every match. Failures (a dir still in use) are ignored.
    """
    root = tempfile.gettempdir()
    try:
        names = os.listdir(root)
    except OSError:
        return
    removed = 0
    for name in names:
        if not name.startswith("tideway-atmos-"):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    if removed:
        log.info("atmos cache: removed %d legacy temp dirs", removed)



def _local_file_is_atmos(path: str) -> bool:
    """True when libav's codec profile for `path` names Dolby Atmos.

    libav reports the JOC object layer as the profile string ("Dolby
    Digital Plus + Dolby Atmos"), which is far more reliable than
    scanning the bitstream — a 'joc' byte scan matched media payload
    bytes in testing. A file whose container doesn't decorate the
    profile stays on the PCM engine: there is no evidence the native
    pipeline would add anything.
    """
    try:
        import av

        with av.open(path) as container:
            for stream in container.streams.audio:
                profile = str(stream.codec_context.profile or "").lower()
                if "atmos" in profile:
                    return True
    except Exception:
        return False
    return False


def _build_source(
    spec: Union[str, list[str]],
    prefetched: Optional[dict[int, bytes]] = None,
    sink: Optional[object] = None,
) -> Union[str, SegmentReader]:
    """Materialize a source spec into something Decoder can open.
    If the caller already fetched the first few segments via the
    byte-level prefetch path, hand them to SegmentReader so it
    skips the corresponding network fetches. `sink`, when given,
    receives every segment byte in order (disk cache tee)."""
    if isinstance(spec, str):
        return spec
    return SegmentReader(spec, prefetched=prefetched, sink=sink)


def _probe_local_stream_info(
    path: str,
) -> tuple[Optional[StreamInfo], Optional[float]]:
    """Read codec/duration metadata for a local file. Returns
    `(info, duration_seconds)`; either or both can be None when
    mutagen can't read the file.

    The duration is what makes seek work on local files. Before
    this returned just the StreamInfo, `_resolve_source` passed
    `None` for duration to the player, which left
    `_current_duration_ms = 0` and made `PCMPlayer.seek` bail at
    its `duration_ms <= 0` guard — the user dragged the scrubber
    and the audio kept playing wherever it was."""
    try:
        from mutagen import File as MutagenFile  # type: ignore
    except Exception:
        return None, None
    try:
        m = MutagenFile(path)
    except Exception:
        return None, None
    if m is None or getattr(m, "info", None) is None:
        return None, None
    info = m.info
    codec: Optional[str] = None
    raw_codec_attr = getattr(info, "codec", None)
    if raw_codec_attr:
        codec = _normalize_codec(raw_codec_attr)
    if codec is None:
        mime_list = getattr(info, "mime", None) or []
        if mime_list:
            codec = _normalize_codec(mime_list[0])
    if codec is None:
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        codec = _normalize_codec(ext)
    is_atmos_local = codec in ("eac3", "ac4") or getattr(info, "channels", 2) > 2
    rg = _read_local_replaygain(m)
    raw_length = getattr(info, "length", None)
    try:
        duration_s = float(raw_length) if raw_length else None
    except (TypeError, ValueError):
        duration_s = None
    stream_info = StreamInfo(
        source="local",
        codec=codec,
        bit_depth=_safe_int(getattr(info, "bits_per_sample", None)),
        sample_rate_hz=_safe_int(getattr(info, "sample_rate", None)),
        audio_quality=(
            getattr(info, "audio_quality", None)
            if hasattr(info, "audio_quality")
            else None
        ),
        audio_mode="DOLBY_ATMOS" if is_atmos_local else "STEREO",
        track_replay_gain_db=rg.track_gain_db,
        track_peak=rg.track_peak,
        album_replay_gain_db=rg.album_gain_db,
        album_peak=rg.album_peak,
    )
    return stream_info, duration_s


def _read_local_replaygain(m) -> ReplayGainTags:
    """Pull ReplayGain values out of mutagen tags so downloaded
    tracks get the same loudness-leveling treatment as streamed ones.

    Standard tag names per format:
      - FLAC / Vorbis: REPLAYGAIN_TRACK_GAIN, REPLAYGAIN_TRACK_PEAK,
        REPLAYGAIN_ALBUM_GAIN, REPLAYGAIN_ALBUM_PEAK (uppercase).
      - MP3 (ID3): TXXX:replaygain_track_gain etc. (case-insensitive).
      - MP4 (iTunes-style): ----:com.apple.iTunes:replaygain_track_gain.

    Mutagen normalises tag access enough that a flat dict-style lookup
    covers FLAC + Vorbis + MP4 in one path. ID3 needs explicit TXXX
    frame iteration. Anything we can't parse falls through to an
    all-None tags object — the resolver treats that as "no data" and
    skips the gain stage for the track, matching streaming behaviour.
    """
    tags: ReplayGainTags = ReplayGainTags()
    raw_tags = getattr(m, "tags", None)
    if raw_tags is None:
        return tags

    def _coerce_gain(value: object) -> Optional[float]:
        if value is None:
            return None
        # Most formats store "-3.45 dB". Strip the unit and parse.
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        s = str(value).strip()
        if not s:
            return None
        if s.lower().endswith(" db"):
            s = s[:-3].strip()
        elif s.lower().endswith("db"):
            s = s[:-2].strip()
        try:
            return float(s)
        except ValueError:
            return None

    def _coerce_peak(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        try:
            return float(str(value).strip())
        except ValueError:
            return None

    # FLAC / Vorbis / MP4 — uppercase keys via mutagen's dict-like
    # tag accessor. Case-insensitive lookup falls out of trying both.
    def _get(key: str) -> object:
        for k in (key, key.upper(), key.lower()):
            try:
                v = raw_tags.get(k) if hasattr(raw_tags, "get") else raw_tags[k]
            except (KeyError, TypeError):
                v = None
            if v is not None:
                return v
        return None

    tags.track_gain_db = _coerce_gain(_get("replaygain_track_gain"))
    tags.track_peak = _coerce_peak(_get("replaygain_track_peak"))
    tags.album_gain_db = _coerce_gain(_get("replaygain_album_gain"))
    tags.album_peak = _coerce_peak(_get("replaygain_album_peak"))

    # ID3 path: scan TXXX frames if we didn't find anything above. MP3s
    # written by various taggers use slightly different casing (Tidal's
    # FLAC→MP3 converters, lame --replaygain-fast, foobar2000) but the
    # frame description is always one of these four canonical strings.
    if tags.track_gain_db is None and hasattr(raw_tags, "getall"):
        try:
            txxx_frames = raw_tags.getall("TXXX") or []
        except Exception:
            txxx_frames = []
        for frame in txxx_frames:
            desc = (getattr(frame, "desc", "") or "").lower()
            text = getattr(frame, "text", None)
            value = text[0] if isinstance(text, (list, tuple)) and text else text
            if desc == "replaygain_track_gain" and tags.track_gain_db is None:
                tags.track_gain_db = _coerce_gain(value)
            elif desc == "replaygain_track_peak" and tags.track_peak is None:
                tags.track_peak = _coerce_peak(value)
            elif desc == "replaygain_album_gain" and tags.album_gain_db is None:
                tags.album_gain_db = _coerce_gain(value)
            elif desc == "replaygain_album_peak" and tags.album_peak is None:
                tags.album_peak = _coerce_peak(value)

    return tags

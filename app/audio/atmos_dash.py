"""Serve a streamed Atmos track to MediaPlayer as DASH.

MediaPlayer cannot start on a file source that is still being written:
given a growing file (or an HTTP response whose body is incomplete) it
stays in Opening/Buffering until the whole source has arrived. That is
why the earlier growing-file prototype played silence. What it *can*
stream is an adaptive source — `AdaptiveMediaSource`, the DASH client
Edge uses for Netflix-style playback — which pulls segments itself and
starts as soon as the first one is buffered.

So the same TIDAL segments the whole-file path joins into one m4a are
published here as a tiny DASH presentation: a manifest, the init
segment, and one file per media segment. A segment request blocks
until that segment has actually downloaded, which is exactly the
back-pressure a streaming client expects; a request for a segment the
sequential pass has not reached yet — every seek — is pulled out of
order instead of waiting for everything before it. The manifest is
static because TIDAL uses a fixed 4 s segment grid (verified across
every cached track: 192000 ticks at timescale 48000).

Nothing here is reachable off the machine: the listener binds
127.0.0.1 on an ephemeral port and each publication is known only by
a random token.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

log = logging.getLogger(__name__)

CHUNK_BYTES = 64 * 1024
# How long a segment request waits for its bytes before giving up. The
# download is expected to stay well ahead of playback; this only caps
# the damage when the network dies mid-track.
_WAIT_S = 90.0
_POLL_S = 0.02

_URL_PREFIX = "/dash/"

_TIMESCALE = 48000
_SEGMENT_UNITS = 192000  # 4.000 s at 48 kHz, TIDAL's fMP4 segment grid


@dataclass
class DashPublication:
    """One track presented as DASH. Everything is written per segment
    so a reader can take segments 1..n while n+1 downloads."""

    token: str
    track_id: str
    directory: str
    n_segments: int
    manifest: bytes
    media_type: str
    init_ready: threading.Event = field(default_factory=threading.Event)
    segments: dict[int, threading.Event] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    failed: bool = False
    # Supplied by the downloader: pull one segment out of order. A seek
    # lands wherever the user dropped the handle, which is usually well
    # ahead of the sequential pass; this is what answers that request
    # without waiting for every segment in between.
    fetch: Optional[Callable[[int], Optional[bytes]]] = field(
        default=None, repr=False
    )
    # Serialises "who delivers segment N": the reader's out-of-order
    # fetch and the sequential pass would otherwise both download it.
    fetch_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )

    def segment_path(self, index: int) -> str:
        import os

        return os.path.join(self.directory, f"seg-{index}.m4s")

    def init_path(self) -> str:
        import os

        return os.path.join(self.directory, "init.mp4")

    def local_segment(self, index: int) -> Optional[bytes]:
        """The segment's bytes when they are already on disk. The event
        is only set after the file is closed, so this never returns a
        partially written segment."""
        event = self.segments.get(index)
        if event is None or not event.is_set():
            return None
        try:
            with open(self.segment_path(index), "rb") as fh:
                return fh.read()
        except OSError:
            return None


class DashServer:
    """Loopback HTTP server for DASH publications."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._publications: dict[str, DashPublication] = {}
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # --- public API --------------------------------------------------

    def publish(
        self,
        track_id: str,
        directory: str,
        n_segments: int,
        *,
        duration_s: Optional[float],
        bandwidth: int,
        codecs: str = "ec-3",
        channels: int = 6,
    ) -> tuple[str, DashPublication]:
        """Announce a track. Returns the manifest URL MediaPlayer should
        be given, and the publication the downloader feeds."""
        token = uuid.uuid4().hex
        manifest = _manifest(
            n_segments, duration_s, bandwidth, codecs, channels
        )
        publication = DashPublication(
            token=token,
            track_id=track_id,
            directory=directory,
            n_segments=n_segments,
            manifest=manifest,
            media_type=_media_type(codecs),
            segments={i: threading.Event() for i in range(1, n_segments + 1)},
        )
        with self._lock:
            httpd = self._ensure_started_locked()
            self._publications[token] = publication
            port = httpd.server_address[1]
        return f"http://127.0.0.1:{port}{_URL_PREFIX}{token}/manifest.mpd", publication

    def write_init(self, publication: DashPublication, data: bytes) -> None:
        with open(publication.init_path(), "wb") as fh:
            fh.write(data)
        publication.init_ready.set()

    def write_segment(
        self, publication: DashPublication, index: int, data: bytes
    ) -> bool:
        """Publish one segment. Returns False when it was already there
        — an out-of-order fetch got to it first, the bytes are the same,
        and a reader may already have the file open."""
        event = publication.segments.get(index)
        if event is None:
            return False
        with publication.fetch_lock:
            if event.is_set():
                return False
            with open(publication.segment_path(index), "wb") as fh:
                fh.write(data)
            event.set()
        return True

    def ensure_segment(self, publication: DashPublication, index: int) -> bool:
        """Make one segment ready before the sequential pass reaches it.

        MediaPlayer asks for the segment under the seek target and
        waits on that response; the sequential pass may be minutes away
        from it, which is what made seeking crawl. Fetching just that
        segment turns the wait into one request.
        """
        event = publication.segments.get(index)
        if event is None:
            return False
        if event.is_set():
            return True
        fetch = publication.fetch
        if fetch is None:
            return False
        with publication.fetch_lock:
            if event.is_set():
                return True
            if publication.done.is_set():
                return False
            try:
                data = fetch(index)
            except Exception as exc:
                log.debug(
                    "atmos dash: out-of-order fetch of segment %d failed: %s",
                    index,
                    exc,
                )
                return False
        if not data:
            return False
        self.write_segment(publication, index, data)
        return True

    def note_done(self, publication: DashPublication, *, failed: bool = False) -> None:
        """The downloader is finished. Pending readers are released so
        a dead stream fails instead of hanging."""
        publication.failed = failed
        publication.done.set()

    def retire(self, publication: Optional[DashPublication]) -> None:
        if publication is None:
            return
        with self._lock:
            self._publications.pop(publication.token, None)
        publication.done.set()

    def is_published_directory(self, directory: str) -> bool:
        with self._lock:
            return any(
                entry.directory == directory
                for entry in self._publications.values()
            )

    def close(self) -> None:
        with self._lock:
            publications = list(self._publications.values())
            self._publications.clear()
            httpd = self._httpd
            self._httpd = None
            thread = self._thread
            self._thread = None
        for publication in publications:
            publication.done.set()
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:
                pass
            try:
                httpd.server_close()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # --- internals ---------------------------------------------------

    def _ensure_started_locked(self) -> ThreadingHTTPServer:
        if self._httpd is not None:
            return self._httpd
        httpd = _Server(("127.0.0.1", 0), _Handler, owner=self)
        thread = threading.Thread(
            target=httpd.serve_forever,
            name="atmos-dash-server",
            daemon=True,
        )
        thread.start()
        self._httpd = httpd
        self._thread = thread
        log.info(
            "atmos dash server listening on 127.0.0.1:%d",
            httpd.server_address[1],
        )
        return httpd

    def lookup(self, token: str) -> Optional[DashPublication]:
        with self._lock:
            return self._publications.get(token)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, owner: DashServer, **kwargs) -> None:
        self.owner = owner
        super().__init__(*args, **kwargs)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # The DASH client re-requests freely; one line per request
        # would bury the real log lines.
        return

    def send_error(self, code, message=None, explain=None) -> None:
        # MediaPlayer drops requests it no longer needs (a seek, a new
        # track), and the wait for a segment only notices when it tries
        # to answer. A dead socket is the client's business, not an
        # error worth a traceback on our stderr.
        try:
            super().send_error(code, message, explain)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(head_only=True)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(head_only=False)

    def _serve(self, *, head_only: bool) -> None:
        owner = getattr(self.server, "owner", None)
        publication = None
        name = ""
        if owner is not None and self.path.startswith(_URL_PREFIX):
            rest = self.path[len(_URL_PREFIX):].split("?", 1)[0]
            token, _, name = rest.partition("/")
            if token and name:
                publication = owner.lookup(token)
        if publication is None or not name:
            self.send_error(404)
            return

        if name.endswith("manifest.mpd"):
            self._send_bytes(publication.manifest, "application/dash+xml", head_only)
            return

        if name == "init.mp4":
            if not self._await(publication.init_ready, publication):
                return
            body = _read(publication.init_path())
            if body is None:
                self.send_error(404)
                return
            self._send_bytes(body, publication.media_type, head_only)
            return

        if name.startswith("seg-") and name.endswith(".m4s"):
            try:
                index = int(name[4:-4])
            except ValueError:
                self.send_error(400)
                return
            event = publication.segments.get(index)
            if event is None:
                self.send_error(404)
                return
            if not event.is_set() and owner is not None:
                # A seek landed past the sequential pass. Pull this one
                # segment now rather than block until the pass arrives.
                owner.ensure_segment(publication, index)
            if not self._await(event, publication):
                return
            body = _read(publication.segment_path(index))
            if body is None:
                self.send_error(404)
                return
            self._send_bytes(body, publication.media_type, head_only)
            return

        self.send_error(404)

    def _await(self, event: threading.Event, publication: DashPublication) -> bool:
        deadline = time.monotonic() + _WAIT_S
        while not event.is_set():
            if publication.done.is_set() or time.monotonic() > deadline:
                self.send_error(404)
                return False
            time.sleep(_POLL_S)
        return True

    def _send_bytes(self, body: bytes, content_type: str, head_only: bool) -> None:
        start = _range_start(self.headers.get("Range"))
        if start > len(body):
            start = 0
        try:
            if start:
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}"
                )
            else:
                self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body) - start))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if head_only:
                return
            self.wfile.write(body[start:])
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        self.close_connection = True


def _read(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _range_start(value: Optional[str]) -> int:
    if not value or not value.startswith("bytes="):
        return 0
    spec = value[len("bytes="):].split(",", 1)[0].strip()
    start = spec.split("-", 1)[0].strip()
    try:
        return max(0, int(start))
    except ValueError:
        return 0


def _media_type(codecs: str) -> str:
    return "audio/mp4"


def _manifest(
    n_segments: int,
    duration_s: Optional[float],
    bandwidth: int,
    codecs: str,
    channels: int,
) -> bytes:
    """A single-representation, static DASH manifest for a track.

    The segment grid is TIDAL's own (4 s, 192000 ticks at 48 kHz), so
    a constant `duration` says the same thing as a timeline and keeps
    Media Foundation's parser on its happy path.
    """
    if duration_s and duration_s > 0:
        presentation = f"PT{duration_s:.3f}S"
    else:
        presentation = f"PT{n_segments * 4}S"
    bandwidth = max(16000, int(bandwidth))
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" profiles="urn:mpeg:dash:profile:isoff-live:2011" type="static" mediaPresentationDuration="{presentation}" minBufferTime="PT2S">
  <Period>
    <AdaptationSet mimeType="audio/mp4" segmentAlignment="true" startWithSAP="1">
      <Representation id="atmos" codecs="{codecs}" bandwidth="{bandwidth}" audioSamplingRate="{_TIMESCALE}">
        <AudioChannelConfiguration schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011" value="{channels}"/>
        <SegmentTemplate timescale="{_TIMESCALE}" duration="{_SEGMENT_UNITS}" startNumber="1" initialization="init.mp4" media="seg-$Number$.m4s"/>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
"""
    return xml.encode("utf-8")


# One server for the process: the listener and thread are paid once,
# and each track publication is just an entry in its map.
dash_server = DashServer()

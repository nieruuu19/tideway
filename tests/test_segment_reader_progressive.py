"""Progressive (byte-range) reads for single-URL stream sources.

The Low / High / Max AAC manifests name one faststart MP4, not a
segment list. Reading that URL as a single "segment" downloaded the
whole 5-10 MB file before `av.open` returned, which is why a Low-tier
track took seconds to start. `SegmentReader` instead fetches fixed
256 KB ranges: the first one carries the moov box, so the decoder
starts immediately, and the rest streams in behind it.

These tests pin the range protocol against a real HTTP server that
honours Range: the first chunk is the only thing on the constructor's
path, a warm seed is continued rather than refetched, a full seed
finishes on 416, and a server that ignores Range is sliced instead of
duplicated.
"""
from __future__ import annotations

import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.audio.segment_reader import SegmentReader
from app.audio.stream_cache import StreamCache

CHUNK = 256 * 1024
PAYLOAD = bytes(range(256)) * (16 * 1024)  # 4 MiB: exactly 16 chunks


class _RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        srv = self.server
        header = self.headers.get("Range")
        start, end = 0, len(PAYLOAD) - 1
        if header:
            match = re.match(r"bytes=(\d+)-(\d+)", header)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
        with srv.lock:  # type: ignore[attr-defined]
            index = len(srv.ranges)  # type: ignore[attr-defined]
            srv.ranges.append(header)  # type: ignore[attr-defined]
        try:
            if index > 0 and srv.delay:  # type: ignore[attr-defined]
                time.sleep(srv.delay)  # type: ignore[attr-defined]
            if srv.ignore_range:  # type: ignore[attr-defined]
                self._send(200, PAYLOAD, None)
            elif start >= len(PAYLOAD):
                self._send(416, b"", f"bytes */{len(PAYLOAD)}")
            else:
                body = PAYLOAD[start:min(end + 1, len(PAYLOAD))]
                self._send(
                    206,
                    body,
                    f"bytes {start}-{start + len(body) - 1}/{len(PAYLOAD)}",
                )
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send(self, status: int, body: bytes, content_range) -> None:
        with self.server.lock:  # type: ignore[attr-defined]
            self.server.statuses.append(status)  # type: ignore[attr-defined]
        self.send_response(status)
        if content_range:
            self.send_header("Content-Range", content_range)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        return


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    srv.daemon_threads = True
    srv.lock = threading.Lock()  # type: ignore[attr-defined]
    srv.ranges: list[str | None] = []  # type: ignore[attr-defined]
    srv.statuses: list[int] = []  # type: ignore[attr-defined]
    srv.delay = 0.0  # type: ignore[attr-defined]
    srv.ignore_range = False  # type: ignore[attr-defined]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    host, port = srv.server_address
    srv.url = f"http://{host}:{port}/track.m4a"  # type: ignore[attr-defined]
    yield srv
    srv.shutdown()
    srv.server_close()


def _wait_for_ranges(server, count: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(server.ranges) >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"only {len(server.ranges)} requests arrived")


def test_the_constructor_returns_after_the_first_chunk(server):
    server.delay = 0.25  # every range after the first is slow

    t0 = time.monotonic()
    reader = SegmentReader([server.url])
    elapsed = time.monotonic() - t0

    assert elapsed < 0.2
    assert len(reader._buf) == CHUNK  # type: ignore[attr-defined]
    assert reader._progressive_done is False  # type: ignore[attr-defined]
    assert server.ranges[0] == f"bytes=0-{CHUNK - 1}"
    reader.close()


def test_read_waits_for_the_ranges_it_needs(server):
    reader = SegmentReader([server.url])

    data = reader.read(CHUNK * 2 + 5)

    assert data == PAYLOAD[: CHUNK * 2 + 5]
    reader.close()


def test_a_range_ignoring_server_is_sliced_not_duplicated(server):
    server.ignore_range = True

    reader = SegmentReader([server.url])
    data = reader.read(-1)

    assert data == PAYLOAD
    assert len(reader._buf) == len(PAYLOAD)  # type: ignore[attr-defined]
    reader.close()


def test_a_range_ignoring_response_continues_a_warm_seed(server):
    server.ignore_range = True
    seed = PAYLOAD[:CHUNK]

    reader = SegmentReader([server.url], prefetched={0: seed})
    data = reader.read(-1)

    assert data == PAYLOAD
    reader.close()


def test_a_warm_seed_continues_where_it_stopped(server):
    seed = PAYLOAD[:CHUNK]

    reader = SegmentReader([server.url], prefetched={0: seed})
    _wait_for_ranges(server, 1)

    # The seed is a prefix of the file, not a whole segment: the next
    # range must start after it, not refetch it.
    assert server.ranges[0] == f"bytes={CHUNK}-{CHUNK * 2 - 1}"
    assert reader.read(-1) == PAYLOAD
    reader.close()


def test_a_seed_holding_the_whole_file_ends_on_416(server):
    reader = SegmentReader([server.url], prefetched={0: PAYLOAD})

    data = reader.read(-1)

    assert data == PAYLOAD
    assert 416 in server.statuses
    assert reader._progressive_done is True  # type: ignore[attr-defined]
    reader.close()


def test_the_disk_cache_gets_the_whole_file_in_one_write(server, tmp_path):
    cache = StreamCache(str(tmp_path))
    sink = cache.sink(
        "555", "low_320k", 1, 180.0, {"source": "stream"}, {"title": "Song"}
    )

    reader = SegmentReader([server.url], sink=sink)
    assert reader.read(-1) == PAYLOAD
    reader.close()

    hit = cache.lookup("555", "low_320k")
    assert hit is not None
    path, duration, info, meta = hit
    assert Path(path).read_bytes() == PAYLOAD
    assert duration == 180.0
    assert info["source"] == "stream"
    assert meta == {"title": "Song"}


def test_close_aborts_the_download_and_drops_the_cache(server, tmp_path):
    server.delay = 0.6
    cache = StreamCache(str(tmp_path))
    sink = cache.sink("556", "low_320k", 1, 180.0, {"source": "stream"}, None)
    reader = SegmentReader([server.url], sink=sink)

    t0 = time.monotonic()
    reader.close()
    elapsed = time.monotonic() - t0

    assert elapsed < 0.3
    assert cache.lookup("556", "low_320k") is None
    assert os.listdir(tmp_path) == []


def test_eof_is_clean_after_the_last_chunk(server):
    reader = SegmentReader([server.url])

    assert reader.read(-1) == PAYLOAD
    assert reader.read(1024) == b""
    reader.close()

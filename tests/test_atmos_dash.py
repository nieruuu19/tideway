"""The loopback DASH presentation MediaPlayer streams while a track
downloads.

MediaPlayer only starts early on an adaptive source, so the segment
server's contract is what decides whether a click is silent: the
manifest must describe TIDAL's own segment grid, a segment request must
block until that segment exists, and a retired or failed download must
fail fast instead of hanging the player.
"""
from __future__ import annotations

import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app.audio.atmos_dash import DashServer


@pytest.fixture()
def server(tmp_path):
    srv = DashServer()
    yield srv, tmp_path
    srv.close()


def _url_parts(url: str) -> tuple[str, str]:
    match = re.match(r"(http://127\.0\.0\.1:\d+/dash/[0-9a-f]+)/(.+)$", url)
    assert match is not None, url
    return match.group(1), match.group(2)


def _get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_manifest_describes_one_ec3_representation_on_tidals_grid(server, tmp_path):
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 54, duration_s=215.36, bandwidth=769000
    )
    manifest = publication.manifest.decode("utf-8")

    assert 'type="static"' in manifest
    assert 'mediaPresentationDuration="PT215.360S"' in manifest
    assert 'codecs="ec-3"' in manifest
    assert 'audioSamplingRate="48000"' in manifest
    # TIDAL's grid: 4 s segments, 192000 ticks at 48 kHz, one-based.
    assert 'timescale="48000"' in manifest
    assert 'duration="192000"' in manifest
    assert 'startNumber="1"' in manifest
    assert 'initialization="init.mp4"' in manifest
    assert 'media="seg-$Number$.m4s"' in manifest
    # Six channels: the E-AC-3 core bed, whatever the JOC layer adds.
    assert 'value="6"' in manifest

    base, name = _url_parts(url)
    status, body = _get(url)
    assert (status, name) == (200, "manifest.mpd")
    assert body == publication.manifest


def test_a_segment_request_waits_for_the_writer_then_serves_it(server, tmp_path):
    """The back-pressure a streaming client expects: asking for a
    segment that has not landed yet blocks, and returns the bytes once
    it does — not an error and not a short body."""
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 3, duration_s=12.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    srv.write_init(publication, b"INIT")
    srv.write_segment(publication, 1, b"seg1")

    assert _get(f"{base}/init.mp4") == (200, b"INIT")
    assert _get(f"{base}/seg-1.m4s") == (200, b"seg1")

    held: list[tuple[int, bytes]] = []
    reader = threading.Thread(
        target=lambda: held.append(_get(f"{base}/seg-2.m4s")), daemon=True
    )
    reader.start()
    time.sleep(0.2)
    assert held == []
    assert not publication.segments[2].is_set()

    srv.write_segment(publication, 2, b"seg2")
    reader.join(timeout=5.0)
    assert held == [(200, b"seg2")]


def test_a_segment_past_the_end_is_a_404(server, tmp_path):
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 2, duration_s=8.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    srv.write_init(publication, b"INIT")

    assert _get(f"{base}/seg-3.m4s")[0] == 404
    assert _get(f"{base}/nothing.mp4")[0] == 404


def test_an_unknown_token_is_a_404(server, tmp_path):
    srv, _directory = server
    url, _publication = srv.publish(
        "123", str(tmp_path), 1, duration_s=4.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    dead = base.rsplit("/", 1)[0] + "/" + "0" * 32

    assert _get(f"{dead}/manifest.mpd")[0] == 404


def test_a_failed_download_releases_blocked_requests(server, tmp_path):
    """A segment request must not hang the player when the download
    died: `note_done(failed=True)` turns the wait into a 404."""
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 3, duration_s=12.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    srv.write_init(publication, b"INIT")

    held: list[tuple[int, bytes]] = []
    reader = threading.Thread(
        target=lambda: held.append(_get(f"{base}/seg-2.m4s")), daemon=True
    )
    reader.start()
    time.sleep(0.2)
    assert held == []

    srv.note_done(publication, failed=True)
    reader.join(timeout=5.0)
    assert held[0][0] == 404


def test_retiring_a_publication_stops_serving_it(server, tmp_path):
    srv, directory = server
    other = tmp_path / "other"
    other.mkdir()
    url, publication = srv.publish(
        "123", str(directory), 1, duration_s=4.0, bandwidth=800000
    )
    url2, publication2 = srv.publish(
        "456", str(other), 1, duration_s=4.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    base2, _name2 = _url_parts(url2)
    srv.write_init(publication, b"INIT")
    srv.write_init(publication2, b"INIT")

    assert srv.is_published_directory(str(directory))
    assert srv.is_published_directory(str(other))

    srv.retire(publication)

    assert not srv.is_published_directory(str(directory))
    assert srv.is_published_directory(str(other))
    assert _get(f"{base}/manifest.mpd")[0] == 404
    # The other track keeps streaming: retiring one must not disturb
    # the one the player is using.
    assert _get(f"{base2}/manifest.mpd")[0] == 200
    # Retiring twice (a stop after a track change) is harmless.
    srv.retire(publication)
    srv.retire(None)


def test_a_head_request_reports_the_size_without_a_body(server, tmp_path):
    """Media Foundation probes sources with HEAD; the reply has to
    carry the length it would get from a GET."""
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 1, duration_s=4.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    srv.write_init(publication, b"INIT")

    request = urllib.request.Request(f"{base}/init.mp4", method="HEAD")
    with urllib.request.urlopen(request, timeout=5.0) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Length"] == "4"
        assert resp.read() == b""


def test_a_seek_segment_is_fetched_instead_of_awaited(server, tmp_path):
    """A seek asks for the segment under the drop point. The sequential
    pass is nowhere near it, so the server pulls that one segment out
    of order — the request must not wait for everything before it."""
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 20, duration_s=80.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    srv.write_init(publication, b"INIT")
    srv.write_segment(publication, 1, b"seg1")
    fetched: list[int] = []

    def fetch(index: int) -> bytes:
        fetched.append(index)
        return f"seg{index}".encode()

    publication.fetch = fetch

    started = time.monotonic()
    assert _get(f"{base}/seg-12.m4s") == (200, b"seg12")
    assert time.monotonic() - started < 2.0
    assert fetched == [12]
    # Nothing between the pass and the target was pulled.
    assert not publication.segments[11].is_set()
    assert Path(directory / "seg-12.m4s").read_bytes() == b"seg12"


def test_a_segment_fetched_for_a_seek_is_not_written_twice(server, tmp_path):
    """The sequential pass reaches the seek segment later; overwriting
    it would race a reader that already has the file open."""
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 20, duration_s=80.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    srv.write_init(publication, b"INIT")
    publication.fetch = lambda index: b"ahead"

    assert _get(f"{base}/seg-9.m4s") == (200, b"ahead")

    assert srv.write_segment(publication, 9, b"late") is False
    assert Path(directory / "seg-9.m4s").read_bytes() == b"ahead"


def test_a_failed_seek_fetch_still_waits_for_the_writer(server, tmp_path):
    """A dead network on the out-of-order request must not cost the
    player the fragment: the request falls back to the sequential
    pass, which is what would have happened without the fast path."""
    srv, directory = server
    url, publication = srv.publish(
        "123", str(directory), 4, duration_s=16.0, bandwidth=800000
    )
    base, _name = _url_parts(url)
    srv.write_init(publication, b"INIT")
    publication.fetch = lambda index: None

    held: list[tuple[int, bytes]] = []
    reader = threading.Thread(
        target=lambda: held.append(_get(f"{base}/seg-3.m4s")), daemon=True
    )
    reader.start()
    time.sleep(0.2)
    assert held == []

    srv.write_segment(publication, 3, b"seg3")
    reader.join(timeout=5.0)
    assert held == [(200, b"seg3")]

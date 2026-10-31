"""Endpoint test for the Other-versions shelf on /api/album/{id}.

The grouping itself is covered in `test_album_versions.py`; this pins
the wiring — that the album endpoint surfaces the group as
`other_versions`, keeps the same editions out of `more_by_artist`, and
still returns a normal payload when the artist has no other editions.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


class StubRelease:
    """Minimal album surface for the endpoint + `_split_album_versions`."""

    def __init__(
        self,
        aid: int,
        name: str,
        *,
        date: str | None = "1969-09-26",
        similar: list | None = None,
    ):
        self.id = aid
        self.name = name
        self.release_date = date
        self.artist = SimpleNamespace(id=10, name="The Beatles")
        self.artists = [self.artist]
        self._similar = similar or []

    def similar(self):
        return self._similar

    def review(self):
        return None


class StubPrimaryArtist:
    id = 10
    name = "The Beatles"

    def __init__(self, releases):
        self._releases = releases

    def get_ep_singles(self, limit=None):
        return []

    def get_other(self, limit=None):
        return []

    def get_similar(self):
        return []

    def __getattr__(self, name):
        raise AssertionError(f"unexpected artist call: {name}")


@pytest.fixture(autouse=True)
def _clear_detail_cache():
    import server

    with server._detail_cache_lock:
        server._detail_cache.clear()
    yield
    with server._detail_cache_lock:
        server._detail_cache.clear()


@pytest.fixture
def client(monkeypatch):
    import server

    releases: list[StubRelease] = []
    current = {}

    monkeypatch.setattr(server, "_require_auth", lambda: None)
    monkeypatch.setattr(
        server,
        "album_to_dict",
        lambda a: {
            "kind": "album",
            "id": str(a.id),
            "name": a.name,
            "num_tracks": 0,
            "media_tags": [],
        },
    )
    monkeypatch.setattr(
        server,
        "track_to_dict",
        lambda t: {"id": str(t.id), "name": t.name},
    )
    monkeypatch.setattr(
        server.tidal, "session", SimpleNamespace(album=lambda _id: current["album"])
    )
    monkeypatch.setattr(server.tidal, "get_album_tracks", lambda _a: [])
    monkeypatch.setattr(server.tidal, "get_artist_albums", lambda _a: releases)

    def _use(album: StubRelease, all_releases: list[StubRelease]):
        current["album"] = album
        releases[:] = all_releases
        current["album"].artist = StubPrimaryArtist(all_releases)

    with TestClient(server.app) as c:
        yield c, _use


def test_other_versions_are_surfaced_and_kept_out_of_more_by(client):
    c, use = client
    current = StubRelease(1, "Abbey Road (2019 Mix)")
    use(
        current,
        [
            current,
            StubRelease(2, "Abbey Road (Super Deluxe Edition)", date="2019-09-27"),
            StubRelease(3, "Abbey Road - 2009 Remaster", date="2009-09-09"),
            StubRelease(4, "Let It Be", date="1970-05-08"),
        ],
    )

    body = c.get("/api/album/1").json()

    assert [a["id"] for a in body["other_versions"]] == ["3", "2"]
    assert [a["id"] for a in body["more_by_artist"]] == ["4"]


def test_album_without_other_editions_gets_an_empty_list(client):
    c, use = client
    current = StubRelease(1, "Abbey Road")
    use(current, [current, StubRelease(4, "Let It Be", date="1970-05-08")])

    body = c.get("/api/album/1").json()

    assert body["other_versions"] == []
    assert [a["id"] for a in body["more_by_artist"]] == ["4"]


def test_similar_albums_backfill_versions_missing_from_the_catalog(client):
    c, use = client
    current = StubRelease(1, "Abbey Road (2019 Mix)")
    use(current, [current, StubRelease(4, "Let It Be", date="1970-05-08")])
    current._similar = [StubRelease(9, "Abbey Road", date="1969-09-26")]

    body = c.get("/api/album/1").json()

    assert [a["id"] for a in body["other_versions"]] == ["9"]


def test_nested_atmos_tag_still_matches_the_standard_release(client):
    c, use = client
    current = StubRelease(1, "Limasawa Street ((Dolby Atmos))")
    use(current, [current, StubRelease(2, "Limasawa Street", date="2019-08-30")])

    body = c.get("/api/album/1").json()

    assert [a["id"] for a in body["other_versions"]] == ["2"]
    assert body["more_by_artist"] == []

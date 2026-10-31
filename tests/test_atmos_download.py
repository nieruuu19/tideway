"""Dolby Atmos download resolution + the `is_atmos` catalog flag.

The downloader's Atmos tier bypasses tidalapi's quality ladder: it
resolves the Atmos catalog twin and pulls the E-AC-3 JOC DASH segments
through app/atmos.py. These tests pin that contract (twin id first,
clicked id as fallback, `.m4a`/`EC-3` result) and the flag that badges
search rows.
"""

import threading
from types import SimpleNamespace

import pytest

from app import downloader


def _build_downloader():
    """Downloader with the worker thread and real session bypassed."""
    dl = downloader.Downloader.__new__(downloader.Downloader)
    dl.tidal = SimpleNamespace(session=SimpleNamespace())
    dl.quality_lock = threading.Lock()
    dl._rate_limit_until = 0.0
    dl._rate_limit_lock = threading.Lock()
    dl._cancelled_ids = set()
    dl._cancelled_lock = threading.Lock()
    return dl


STREAM = SimpleNamespace(
    urls=["https://cdn/init.mp4", "https://cdn/seg-1.mp4"],
    codec="ec-3",
    sample_rate=48000,
    channels=6,
)


def test_fetch_atmos_sources_returns_eac3_segments_as_m4a(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(
        downloader.atmos_mod,
        "find_atmos_track_id",
        lambda session, track: 999,
    )

    def fake_fetch(session, track_id):
        seen.append(track_id)
        return STREAM

    monkeypatch.setattr(downloader.atmos_mod, "fetch_atmos_stream", fake_fetch)

    urls, ext, codec = _build_downloader()._fetch_atmos_sources(
        SimpleNamespace(id=123)
    )

    assert urls == STREAM.urls
    assert ext == ".m4a"
    assert codec == "EC-3"
    # The resolved twin is authoritative; don't re-fetch the clicked id.
    assert seen == [999]


def test_fetch_atmos_sources_falls_back_to_clicked_id(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(
        downloader.atmos_mod,
        "find_atmos_track_id",
        lambda session, track: 999,
    )

    def fake_fetch(session, track_id):
        seen.append(track_id)
        if track_id == 999:
            raise downloader.atmos_mod.AtmosUnavailable("no twin asset")
        return STREAM

    monkeypatch.setattr(downloader.atmos_mod, "fetch_atmos_stream", fake_fetch)

    urls, ext, _ = _build_downloader()._fetch_atmos_sources(
        SimpleNamespace(id=123)
    )

    assert urls == STREAM.urls
    assert ext == ".m4a"
    assert seen == [999, 123]


def test_fetch_atmos_sources_raises_when_no_asset(monkeypatch):
    monkeypatch.setattr(
        downloader.atmos_mod,
        "find_atmos_track_id",
        lambda session, track: None,
    )

    def fake_fetch(session, track_id):
        raise downloader.atmos_mod.AtmosUnavailable("no atmos mix")

    monkeypatch.setattr(downloader.atmos_mod, "fetch_atmos_stream", fake_fetch)

    with pytest.raises(downloader.atmos_mod.AtmosUnavailable):
        _build_downloader()._fetch_atmos_sources(SimpleNamespace(id=123))


def _fake_track(**overrides):
    base = dict(
        id="123",
        name="Song",
        duration=100,
        track_num=1,
        explicit=False,
        artists=[],
        album=None,
        share_url=None,
        mixes={},
        media_metadata_tags=[],
        isrc=None,
        ai=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_dolby_atmos": True},
        {"audio_modes": ["DOLBY_ATMOS"]},
        {"media_metadata_tags": ["DOLBY_ATMOS"]},
    ],
)
def test_track_to_dict_marks_atmos(overrides):
    import server

    assert server.track_to_dict(_fake_track(**overrides))["is_atmos"] is True


def test_track_to_dict_is_atmos_false_for_stereo():
    import server

    assert server.track_to_dict(_fake_track())["is_atmos"] is False


@pytest.mark.parametrize("max_quality", ["high_lossless", "hi_res_lossless", None])
def test_qualities_endpoint_always_offers_atmos(monkeypatch, max_quality):
    """The download menu must list Atmos regardless of the stereo tier:
    TIDAL's subscription ceiling tracks FLAC resolution only, and HiFi
    sessions still resolve E-AC-3 JOC manifests."""
    import server

    monkeypatch.setattr(server, "_require_local_access", lambda: None)
    monkeypatch.setattr(server.tidal, "get_max_quality", lambda: max_quality)

    values = [q["value"] for q in server.list_qualities()]

    assert values[0] == "atmos"
    assert "high_lossless" in values


# ---------------------------------------------------------------------------
# Filenames: Atmos must not collide with the stereo edition.
# ---------------------------------------------------------------------------


def _settings(tmp_path, template="{artist} - {title}"):
    return SimpleNamespace(
        filename_template=template,
        output_dir=str(tmp_path),
        create_album_folders=False,
        create_playlist_folders=False,
        album_folder_includes_artist=False,
    )


def _item(quality):
    return downloader.DownloadItem(
        item_id="abc",
        url="",
        title="Song",
        artist="Artist",
        quality=quality,
    )


def test_build_path_suffixes_atmos_downloads(tmp_path):
    path = downloader._build_path(_item("atmos"), _settings(tmp_path), ".m4a")

    assert path.name == "Artist - Song [Dolby Atmos].m4a"


def test_build_path_leaves_other_qualities_untouched(tmp_path):
    path = downloader._build_path(
        _item("hi_res_lossless"), _settings(tmp_path), ".flac"
    )

    assert path.name == "Artist - Song.flac"


def test_build_path_skips_suffix_when_template_encodes_quality(tmp_path):
    settings = _settings(tmp_path, template="{artist} - {title} [{Codec}]")

    path = downloader._build_path(_item("atmos"), settings, ".m4a")

    assert path.name == "Artist - Song [{Codec}].m4a"

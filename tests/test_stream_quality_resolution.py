"""Lossless/Max downloads must not silently land as AAC 320.

TIDAL's legacy `urlpostpaywall` endpoint (what `Track.get_url()`
calls) answers AAC 320 for every tier, and the modern
`playbackinfopostpaywall` path can also downgrade a LOSSLESS request
to AAC on clients that don't serve the CD tier. `request_stream_manifest`
is the single resolver for downloads, streaming, and the HTTP proxy:
these tests pin the escalation / fallback rules and the downloader's
manifest-first fetch.
"""

import threading
from types import SimpleNamespace

import pytest
import tidalapi

from app import tidal_client


class _AuthError(Exception):
    def __init__(self, status_code: int, text: str = ""):
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code, text=text)


def _stream(tier: str):
    """A fake Stream whose audio_quality is the tidalapi Quality enum."""
    return SimpleNamespace(
        audio_quality=tidalapi.Quality(tier),
        get_stream_manifest=lambda: SimpleNamespace(
            urls=[f"https://cdn/{tier}/0"], is_encrypted=False
        ),
    )


def _session(quality: str = "high_lossless"):
    return SimpleNamespace(
        config=SimpleNamespace(quality=tidalapi.Quality[quality])
    )


def _track(session, served, raises=None):
    """A track whose get_stream() answers per the session's current tier.

    `served` maps requested tier name -> tier TIDAL renders; `raises`
    maps requested tier name -> exception to throw instead.
    """

    class Track:
        id = 42
        calls: list[str] = []

        def get_stream(self):
            tier = session.config.quality.name
            self.calls.append(tier)
            if raises and tier in raises:
                raise raises[tier]
            return _stream(served[tier])

    return Track()


# ---------------------------------------------------------------------------
# delivered_quality / is_lossy_quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (tidalapi.Quality.hi_res_lossless, "hi_res_lossless"),
        ("HI_RES_LOSSLESS", "hi_res_lossless"),
        ("high", "low_320k"),
        (None, None),
    ],
)
def test_delivered_quality_maps_enum_and_raw_strings(raw, expected):
    assert (
        tidal_client.delivered_quality(SimpleNamespace(audio_quality=raw))
        == expected
    )


def test_delivered_quality_missing_attribute_is_unknown():
    assert tidal_client.delivered_quality(SimpleNamespace()) is None


def test_is_lossy_quality():
    assert tidal_client.is_lossy_quality("low_320k")
    assert tidal_client.is_lossy_quality("low_96k")
    assert not tidal_client.is_lossy_quality("high_lossless")
    assert not tidal_client.is_lossy_quality(None)


# ---------------------------------------------------------------------------
# request_stream_manifest
# ---------------------------------------------------------------------------


def test_lossless_served_as_aac_escalates_to_hi_res():
    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH", "hi_res_lossless": "HI_RES_LOSSLESS"},
    )

    stream, manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert track.calls == ["high_lossless", "hi_res_lossless"]
    assert delivered == "hi_res_lossless"
    assert stream.audio_quality is tidalapi.Quality.hi_res_lossless
    assert manifest.urls == ["https://cdn/HI_RES_LOSSLESS/0"]
    # The caller's tier is restored; the override was temporary.
    assert session.config.quality is tidalapi.Quality.high_lossless


def test_lossless_accepted_when_tidal_serves_flac():
    session = _session("high_lossless")
    track = _track(session, served={"high_lossless": "LOSSLESS"})

    _s, _m, delivered = tidal_client.request_stream_manifest(session, track)

    assert track.calls == ["high_lossless"]
    assert delivered == "high_lossless"


def test_failed_escalation_keeps_the_original_lossy_stream():
    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH"},
        raises={"hi_res_lossless": RuntimeError("boom")},
    )

    stream, _m, delivered = tidal_client.request_stream_manifest(session, track)

    assert delivered == "low_320k"
    assert stream.audio_quality is tidalapi.Quality("HIGH")
    assert session.config.quality is tidalapi.Quality.high_lossless


def test_still_lossy_escalation_keeps_the_original_stream():
    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH", "hi_res_lossless": "HIGH"},
    )

    _s, _m, delivered = tidal_client.request_stream_manifest(session, track)

    assert delivered == "low_320k"


def test_a_lossy_answer_takes_the_flac_route_without_a_second_v1_call(
    monkeypatch,
):
    """One v1 call, then the manifest route — not v1, v1 again, then
    the manifest. The extra round-trip is what pushed a fast album
    click-through into TIDAL's rate limit."""
    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH", "hi_res_lossless": "HI_RES_LOSSLESS"},
    )
    calls = _patch_flac(monkeypatch, _flac_asset())

    _stream, _manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert track.calls == ["high_lossless"]
    assert calls == [(42, False)]
    assert delivered == "high_lossless"


def test_no_flac_asset_still_escalates_to_hi_res(monkeypatch):
    """The v1 hi-res retry stays as the last resort for clients where
    that tier does render, so dropping the FLAC route must not lose the
    lossless stream."""
    from app.lossless import LosslessUnavailable

    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH", "hi_res_lossless": "HI_RES_LOSSLESS"},
    )
    _patch_flac(monkeypatch, error=LosslessUnavailable("no FLAC asset"))

    _stream, _manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert track.calls == ["high_lossless", "hi_res_lossless"]
    assert delivered == "hi_res_lossless"


def test_lossy_request_is_never_escalated():
    session = _session("low_320k")
    track = _track(session, served={"low_320k": "HIGH"})

    _s, _m, delivered = tidal_client.request_stream_manifest(session, track)

    assert track.calls == ["low_320k"]
    assert delivered == "low_320k"


def test_hi_res_401_falls_back_to_lossless():
    session = _session("hi_res_lossless")
    track = _track(
        session,
        served={"high_lossless": "LOSSLESS"},
        raises={"hi_res_lossless": _AuthError(401)},
    )

    _s, _m, delivered = tidal_client.request_stream_manifest(session, track)

    assert track.calls == ["hi_res_lossless", "high_lossless"]
    assert delivered == "high_lossless"


def test_asset_not_ready_is_not_treated_as_a_token_problem():
    """TIDAL answers 401 subStatus 4005 for a track it can't serve in
    this region. It is a track-level refusal: no lossless retry, no
    session refresh — just a message the UI can show."""
    session = _session("hi_res_lossless")
    not_ready = _AuthError(
        401,
        '{"status":401,"subStatus":4005,'
        '"userMessage":"Asset is not ready for playback"}',
    )
    track = _track(session, served={}, raises={"hi_res_lossless": not_ready})

    with pytest.raises(tidal_client.TidalAssetUnavailable, match="asset not ready"):
        tidal_client.request_stream_manifest(session, track)

    assert track.calls == ["hi_res_lossless"]
    assert session.config.quality is tidalapi.Quality.hi_res_lossless


def test_asset_not_ready_on_lossless_fallback_raises_dedicated_error():
    session = _session("hi_res_lossless")
    track = _track(
        session,
        served={},
        raises={
            "hi_res_lossless": _AuthError(401),
            "high_lossless": _AuthError(
                401, '{"subStatus":4005,"userMessage":"Asset is not ready for playback"}'
            ),
        },
    )

    with pytest.raises(tidal_client.TidalAssetUnavailable, match="asset not ready"):
        tidal_client.request_stream_manifest(session, track)

    assert track.calls == ["hi_res_lossless", "high_lossless"]


def test_non_auth_error_propagates():
    session = _session("hi_res_lossless")

    class Track:
        id = 1

        def get_stream(self):
            raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        tidal_client.request_stream_manifest(session, Track())


# ---------------------------------------------------------------------------
# Downloader._fetch_stream_sources
# ---------------------------------------------------------------------------


def _build_downloader(session=None):
    from app import downloader

    dl = downloader.Downloader.__new__(downloader.Downloader)
    dl.tidal = SimpleNamespace(session=session or SimpleNamespace())
    dl.quality_lock = threading.Lock()
    dl._rate_limit_until = 0.0
    dl._rate_limit_lock = threading.Lock()
    return dl


def _manifest_track(urls, ext=".mp4", codec="flac", encrypted=False):
    manifest = SimpleNamespace(
        urls=list(urls),
        is_encrypted=encrypted,
        file_extension=ext,
        codecs=codec,
    )
    return SimpleNamespace(
        id=7,
        get_stream=lambda: SimpleNamespace(
            audio_quality=tidalapi.Quality("LOSSLESS"),
            get_stream_manifest=lambda: manifest,
        ),
    )


def test_fetch_stream_sources_returns_manifest_and_restores_quality():
    session = SimpleNamespace(
        config=SimpleNamespace(quality=tidalapi.Quality.low_320k)
    )
    dl = _build_downloader(session)
    track = _manifest_track(["https://cdn/0", "https://cdn/1"])

    urls, ext, codec, delivered = dl._fetch_stream_sources(
        track, "hi_res_lossless"
    )

    assert urls == ["https://cdn/0", "https://cdn/1"]
    assert ext == ".mp4"
    assert codec == "FLAC"
    assert delivered == "high_lossless"
    assert session.config.quality is tidalapi.Quality.low_320k


def test_fetch_stream_sources_falls_back_to_get_url_without_manifest():
    session = SimpleNamespace(
        config=SimpleNamespace(quality=tidalapi.Quality.low_320k),
        is_pkce=False,
    )
    dl = _build_downloader(session)

    def no_manifest():
        raise tidalapi.exceptions.StreamNotAvailable("nope")

    track = SimpleNamespace(
        id=7,
        get_stream=no_manifest,
        get_url=lambda: "https://cdn/legacy.aac",
    )

    urls, ext, codec, delivered = dl._fetch_stream_sources(
        track, "high_lossless"
    )

    assert urls == ["https://cdn/legacy.aac"]
    assert ext is None and codec is None and delivered is None


def test_fetch_stream_sources_reraises_for_pkce_sessions():
    session = SimpleNamespace(
        config=SimpleNamespace(quality=tidalapi.Quality.low_320k),
        is_pkce=True,
    )
    dl = _build_downloader(session)

    def no_manifest():
        raise tidalapi.exceptions.StreamNotAvailable("nope")

    track = SimpleNamespace(id=7, get_stream=no_manifest)

    with pytest.raises(tidalapi.exceptions.StreamNotAvailable):
        dl._fetch_stream_sources(track, "high_lossless")


def test_fetch_stream_sources_refuses_encrypted_manifest():
    session = SimpleNamespace(
        config=SimpleNamespace(quality=tidalapi.Quality.low_320k)
    )
    dl = _build_downloader(session)
    track = _manifest_track(["https://cdn/0"], encrypted=True)

    with pytest.raises(RuntimeError, match="encrypted"):
        dl._fetch_stream_sources(track, "high_lossless")


# ---------------------------------------------------------------------------
# OpenAPI FLAC fallback (v1 answered AAC for a lossless request)
# ---------------------------------------------------------------------------


def _flac_asset(hi_res: bool = False):
    from app.lossless import LosslessStream

    return LosslessStream(
        urls=["https://cdn/flac/init.mp4", "https://cdn/flac/seg1.m4s"],
        sample_rate=96000 if hi_res else 44100,
        bit_depth=24 if hi_res else 16,
        channels=2,
        track_replay_gain=-6.5,
        track_peak=0.99,
        album_replay_gain=-5.0,
        album_peak=0.95,
    )


def _patch_flac(monkeypatch, asset=None, error=None):
    from app import lossless

    calls: list[tuple[int, bool]] = []

    def _fetch(_session, track_id, *, hi_res):
        calls.append((track_id, hi_res))
        if error is not None:
            raise error
        return asset

    monkeypatch.setattr(tidal_client.lossless_mod, "fetch_lossless_stream", _fetch)
    return calls


def test_lossless_served_as_aac_takes_the_flac_manifest(monkeypatch):
    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH", "hi_res_lossless": "HIGH"},
    )
    calls = _patch_flac(monkeypatch, _flac_asset())

    stream, manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert calls == [(42, False)]
    assert delivered == "high_lossless"
    assert stream.audio_quality == "LOSSLESS"
    assert stream.bit_depth == 16
    assert stream.sample_rate == 44100
    assert stream.track_replay_gain == -6.5
    assert stream.album_peak_amplitude == 0.95
    assert manifest.urls == ["https://cdn/flac/init.mp4", "https://cdn/flac/seg1.m4s"]
    assert manifest.file_extension == "flac"
    assert manifest.get_codecs() == "flac"
    assert manifest.is_encrypted is False
    # The caller's tier is restored after the escalation attempts.
    assert session.config.quality is tidalapi.Quality.high_lossless


def test_max_request_takes_the_flac_manifest_as_hi_res(monkeypatch):
    session = _session("hi_res_lossless")
    track = _track(session, served={"hi_res_lossless": "HIGH"})
    calls = _patch_flac(monkeypatch, _flac_asset(hi_res=True))

    stream, _manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert calls == [(42, True)]
    assert delivered == "hi_res_lossless"
    assert stream.audio_quality == "HI_RES_LOSSLESS"
    assert stream.bit_depth == 24


def test_a_cd_master_under_a_max_request_is_reported_honestly(monkeypatch):
    """Max asked; the only FLAC TIDAL serves for this master is CD
    rate. The badge must say lossless, not hi-res."""
    session = _session("hi_res_lossless")
    track = _track(session, served={"hi_res_lossless": "HIGH"})
    _patch_flac(monkeypatch, _flac_asset(hi_res=False))

    stream, _manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert delivered == "high_lossless"
    assert stream.audio_quality == "LOSSLESS"


def test_flac_route_failure_keeps_the_aac_stream(monkeypatch):
    from app import lossless

    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH", "hi_res_lossless": "HIGH"},
    )
    _patch_flac(
        monkeypatch, error=lossless.LosslessUnavailable("no FLAC asset")
    )

    stream, _manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert delivered == "low_320k"
    assert stream.audio_quality is tidalapi.Quality("HIGH")


def test_a_broken_session_never_loses_the_aac_stream(monkeypatch):
    """A session without the OpenAPI request helper (older tidalapi, a
    test double) must not turn a working AAC play into an error."""
    session = _session("high_lossless")
    track = _track(
        session,
        served={"high_lossless": "HIGH", "hi_res_lossless": "HIGH"},
    )

    _stream_, _manifest, delivered = tidal_client.request_stream_manifest(
        session, track
    )

    assert delivered == "low_320k"


# ---------------------------------------------------------------------------
# Stale cache entries: AAC stored under a lossless key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quality, raw, usable",
    [
        ("high_lossless", "LOW", False),
        ("high_lossless", "HIGH", False),
        ("hi_res_lossless", tidalapi.Quality("HIGH"), False),
        ("high_lossless", "LOSSLESS", True),
        ("high_lossless", "HI_RES_LOSSLESS", True),
        ("hi_res_lossless", "LOSSLESS", True),
        # Only lossless requests are validated: a Low play may open
        # whatever it cached.
        ("low_320k", "LOW", True),
        ("low_96k", "HIGH", True),
        # Unknown facts are trusted rather than forcing a re-download.
        ("high_lossless", None, True),
        ("high_lossless", SimpleNamespace(), True),
    ],
)
def test_cached_tier_validation(quality, raw, usable):
    from app.audio import player as player_mod

    info = (
        raw
        if isinstance(raw, SimpleNamespace)
        else SimpleNamespace(audio_quality=raw)
    )
    assert player_mod._cached_tier_is_usable(quality, info) is usable


def test_cached_tier_validation_without_quality_accepts_anything():
    from app.audio import player as player_mod

    assert player_mod._cached_tier_is_usable(None, None) is True
    assert player_mod._cached_tier_is_usable("high_lossless", None) is True

"""Tests for Dolby Atmos (E-AC-3 JOC) resolution.

Covers the pieces that don't need a live Tidal session: DASH MPD
expansion, the base64 manifest data-URI, OpenAPI error classification,
catalog "twin" matching, and the player's Atmos -> lossless fallback.
"""
from __future__ import annotations

import base64
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import atmos
from app.audio.manifest_cache import ManifestCache


@pytest.fixture(autouse=True)
def _clear_atmos_caches():
    atmos._TRACK_TWIN_CACHE.clear()
    atmos._ALBUM_TWIN_CACHE.clear()
    yield
    atmos._TRACK_TWIN_CACHE.clear()
    atmos._ALBUM_TWIN_CACHE.clear()


MPD = """<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" mediaPresentationDuration="PT12.5S">
  <BaseURL>https://cdn.example.test/audio/</BaseURL>
  <Period duration="PT12.5S">
    <AdaptationSet contentType="audio" mimeType="audio/mp4" audioSamplingRate="48000">
      <SegmentTemplate timescale="1000"
                       initialization="$RepresentationID$/init.mp4"
                       media="$RepresentationID$/seg-$Number%05d$.mp4">
        <SegmentTimeline>
          <S t="0" d="5000" r="2"/>
        </SegmentTimeline>
      </SegmentTemplate>
      <Representation id="ec3-5_1" codecs="ec-3" bandwidth="448000">
        <AudioChannelConfiguration
            schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011"
            value="6"/>
      </Representation>
      <Representation id="aac-stereo" codecs="mp4a.40.2" bandwidth="256000"
                      audioSamplingRate="44100">
        <AudioChannelConfiguration
            schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011"
            value="2"/>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
"""


def _data_uri(xml: str) -> str:
    return "data:application/dash+xml;base64," + base64.b64encode(
        xml.encode("utf-8")
    ).decode("ascii")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeRequestSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, path, params=None, headers=None, base_url=None):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "params": params or {},
                "headers": headers or {},
                "base_url": base_url,
            }
        )
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return _FakeResponse(value)


class _FakeSession:
    def __init__(self, responses):
        self.config = SimpleNamespace(
            openapi_v2_location="https://openapi.tidal.com/v2/"
        )
        self.request = _FakeRequestSession(responses)


class _HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


def _manifest_payload(xml: str = MPD, **attrs) -> dict:
    attributes = {
        "formats": ["EAC3_JOC"],
        "uri": _data_uri(xml),
        "trackPresentation": "FULL",
        "hash": "abc123",
        "trackAudioNormalizationData": {
            "replayGain": -6.5,
            "peakAmplitude": 0.98,
        },
        "albumAudioNormalizationData": {
            "replayGain": -5.0,
            "peakAmplitude": 0.95,
        },
    }
    attributes.update(attrs)
    return {"data": {"attributes": attributes}}


# ---------------------------------------------------------------------------
# DASH parsing
# ---------------------------------------------------------------------------


def test_parse_dash_expands_timeline_and_base_url_chain():
    tracks = atmos.parse_dash_audio(MPD)
    by_id = {t["id"]: t for t in tracks}

    ec3 = by_id["ec3-5_1"]
    assert ec3["codec"] == "ec-3"
    assert ec3["channels"] == 6
    assert ec3["sampleRate"] == 48000
    assert ec3["urls"] == [
        "https://cdn.example.test/audio/ec3-5_1/init.mp4",
        "https://cdn.example.test/audio/ec3-5_1/seg-00001.mp4",
        "https://cdn.example.test/audio/ec3-5_1/seg-00002.mp4",
        "https://cdn.example.test/audio/ec3-5_1/seg-00003.mp4",
    ]
    assert by_id["aac-stereo"]["channels"] == 2
    assert by_id["aac-stereo"]["sampleRate"] == 44100


def test_best_dash_representation_prefers_eac3_over_listing_order():
    best = atmos.best_dash_representation(MPD)
    assert best["id"] == "ec3-5_1"


def test_negative_repeat_runs_until_next_timestamp():
    xml = """<MPD xmlns="urn:mpeg:dash:schema:mpd:2011">
      <Period duration="PT10S">
        <AdaptationSet contentType="audio" mimeType="audio/mp4">
          <SegmentTemplate timescale="1000" duration="2000"
              initialization="init.mp4" media="$Number$.mp4">
            <SegmentTimeline>
              <S t="0" d="2000" r="1"/>
              <S t="6000" d="2000" r="-1"/>
            </SegmentTimeline>
          </SegmentTemplate>
          <Representation id="r" codecs="ec-3"/>
        </AdaptationSet>
      </Period>
    </MPD>"""
    urls = atmos.parse_dash_audio(xml)[0]["urls"]
    # init + 2 (r=1) + 2 (r=-1 runs to 6s -> 10s period end)
    assert len(urls) == 5
    assert [u.rsplit("/", 1)[-1] for u in urls[1:]] == [
        "1.mp4",
        "2.mp4",
        "3.mp4",
        "4.mp4",
    ]


def test_parse_dash_rejects_drm():
    xml = MPD.replace(
        "<Representation id=\"ec3-5_1\"",
        "<ContentProtection schemeIdUri=\"urn:mpeg:dash:mp4protection:2011\"/>"
        "<Representation id=\"ec3-5_1\"",
    )
    with pytest.raises(atmos.AtmosUnavailable, match="DRM"):
        atmos.parse_dash_audio(xml)


def test_parse_dash_rejects_multi_period():
    xml = MPD.replace(
        "</Period>", '</Period><Period duration="PT1S"></Period>'
    )
    with pytest.raises(atmos.AtmosUnavailable, match="single complete period"):
        atmos.parse_dash_audio(xml)


def test_parse_dash_rejects_missing_timeline():
    xml = """<MPD xmlns="urn:mpeg:dash:schema:mpd:2011">
      <Period>
        <AdaptationSet contentType="audio" mimeType="audio/mp4">
          <SegmentTemplate initialization="init.mp4" media="s-$Number$.mp4"/>
          <Representation id="r" codecs="ec-3"/>
        </AdaptationSet>
      </Period>
    </MPD>"""
    with pytest.raises(atmos.AtmosUnavailable, match="finite segment timeline"):
        atmos.parse_dash_audio(xml)


def test_parse_dash_rejects_wide_number_padding():
    xml = MPD.replace("$Number%05d$", "$Number%099d$")
    with pytest.raises(atmos.AtmosUnavailable, match="width"):
        atmos.parse_dash_audio(xml)


def test_parse_dash_treats_malformed_xml_as_unavailable():
    with pytest.raises(atmos.AtmosUnavailable, match="invalid XML"):
        atmos.parse_dash_audio("<MPD><Period>")


def test_expand_template_escaped_dollar():
    rep = {"id": "r", "bandwidth": "1"}
    assert atmos._expand_template("a$$b-$Number$.mp4", rep, 7, 0) == "a$b-7.mp4"


# ---------------------------------------------------------------------------
# Data URI + error classification
# ---------------------------------------------------------------------------


def test_decode_manifest_data_uri_round_trip():
    assert atmos._decode_manifest_data_uri(_data_uri(MPD)) == MPD


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "https://cdn.example.test/manifest.mpd",
        "data:application/dash+xml;charset=utf-8,<MPD/>",
        "data:application/dash+xml;base64,!!!not-base64!!!",
    ],
)
def test_decode_manifest_data_uri_rejects_garbage(uri):
    with pytest.raises(atmos.AtmosUnavailable):
        atmos._decode_manifest_data_uri(uri)


@pytest.mark.parametrize(
    ("exc", "unavailable"),
    [
        (_HttpError(404), True),
        (_HttpError(403), True),
        (_HttpError(400), True),
        (_HttpError(405), True),
        (_HttpError(401), False),
        (_HttpError(429), False),
        (_HttpError(500), False),
        (RuntimeError("boom"), False),
    ],
)
def test_unavailable_error_classification(exc, unavailable):
    assert atmos._is_unavailable_error(exc) is unavailable


def test_unavailable_error_classification_by_type_name():
    class ObjectNotFound(Exception):
        pass

    assert atmos._is_unavailable_error(ObjectNotFound("nope")) is True


# ---------------------------------------------------------------------------
# fetch_atmos_stream
# ---------------------------------------------------------------------------


def test_fetch_atmos_stream_uses_openapi_manifest_and_normalization():
    session = _FakeSession([_manifest_payload()])
    stream = atmos.fetch_atmos_stream(session, 12345)

    call = session.request.calls[0]
    assert call["path"] == "trackManifests/12345"
    assert call["base_url"] == "https://openapi.tidal.com/v2/"
    assert call["params"]["formats"] == "EAC3_JOC"
    assert call["params"]["manifestType"] == "MPEG_DASH"
    assert call["params"]["usage"] == "PLAYBACK"
    assert call["headers"]["Accept"] == "application/vnd.api+json"

    assert stream.codec == "ec-3"
    assert stream.channels == 6
    assert stream.sample_rate == 48000
    assert stream.manifest_hash == "abc123"
    assert stream.track_replay_gain == pytest.approx(-6.5)
    assert stream.track_peak == pytest.approx(0.98)
    assert stream.album_replay_gain == pytest.approx(-5.0)
    assert len(stream.urls) == 4
    assert stream.urls[0].endswith("ec3-5_1/init.mp4")


def test_fetch_atmos_stream_falls_back_to_download_usage():
    session = _FakeSession([_HttpError(404), _manifest_payload()])
    stream = atmos.fetch_atmos_stream(session, 7)
    assert [c["params"]["usage"] for c in session.request.calls] == [
        "PLAYBACK",
        "DOWNLOAD",
    ]
    assert stream.urls


def test_fetch_atmos_stream_raises_when_both_usages_unavailable():
    session = _FakeSession([_HttpError(404), _HttpError(403)])
    with pytest.raises(atmos.AtmosUnavailable):
        atmos.fetch_atmos_stream(session, 7)


def test_fetch_atmos_stream_propagates_auth_and_rate_limit_errors():
    with pytest.raises(_HttpError) as excinfo:
        atmos.fetch_atmos_stream(_FakeSession([_HttpError(401)]), 7)
    assert excinfo.value.response.status_code == 401

    with pytest.raises(_HttpError):
        atmos.fetch_atmos_stream(_FakeSession([_HttpError(429)]), 7)


def test_fetch_atmos_stream_rejects_preview_and_missing_format():
    # Preview is refused for both usages, so it needs the PLAYBACK and
    # DOWNLOAD responses.
    preview = _manifest_payload(trackPresentation="PREVIEW")
    session = _FakeSession([preview, preview])
    with pytest.raises(atmos.AtmosUnavailable, match="preview"):
        atmos.fetch_atmos_stream(session, 7)
    assert [c["params"]["usage"] for c in session.request.calls] == [
        "PLAYBACK",
        "DOWNLOAD",
    ]

    with pytest.raises(atmos.AtmosUnavailable):
        atmos.fetch_atmos_stream(
            _FakeSession([_manifest_payload(formats=["FLAC"])]), 7
        )


def test_fetch_atmos_stream_requires_eac3_representation():
    aac_only = MPD.replace('codecs="ec-3"', 'codecs="mp4a.40.2"')
    with pytest.raises(atmos.AtmosUnavailable, match="E-AC-3"):
        atmos.fetch_atmos_stream(_FakeSession([_manifest_payload(aac_only)]), 7)


# ---------------------------------------------------------------------------
# Catalog twin lookup
# ---------------------------------------------------------------------------


def test_is_atmos_reads_audio_modes():
    assert atmos.is_atmos(SimpleNamespace(audio_modes=["DOLBY_ATMOS"]))
    assert not atmos.is_atmos(SimpleNamespace(audio_modes=["STEREO"]))
    assert not atmos.is_atmos(SimpleNamespace())


def test_normalize_title_strips_atmos_suffix():
    assert atmos._normalize_title("Random Access Memories (Dolby Atmos)") == (
        "random access memories"
    )
    assert atmos._normalize_title("Discovery - Dolby Atmos") == "discovery"
    assert atmos._normalize_title("  Discovery  ") == "discovery"


def test_find_atmos_track_id_returns_own_id_for_atmos_track():
    track = SimpleNamespace(id=42, audio_modes=["DOLBY_ATMOS"])
    session = MagicMock()
    assert atmos.find_atmos_track_id(session, track) == 42


def test_find_atmos_track_id_resolves_album_twin_by_isrc():
    source_track = SimpleNamespace(
        id=11,
        title="One More Time",
        isrc="USAT29900609",
        track_num=1,
        volume_num=1,
        audio_modes=[],
        album=SimpleNamespace(id=100),
    )
    stereo_album = SimpleNamespace(
        id=100,
        name="Discovery",
        num_tracks=2,
        explicit=False,
        artist=SimpleNamespace(id=555),
    )
    twin_track = SimpleNamespace(
        id=9001,
        title="One More Time",
        isrc="USAT29900609",
        track_num=1,
        volume_num=1,
    )
    twin_album = SimpleNamespace(
        id=200,
        name="Discovery (Dolby Atmos)",
        num_tracks=2,
        explicit=False,
        artist=SimpleNamespace(id=555),
        tracks=lambda: [twin_track],
    )
    twin_album.audio_modes = ["DOLBY_ATMOS"]

    session = MagicMock()
    session.album.return_value = stereo_album
    artist = MagicMock()
    artist.get_albums.return_value = [twin_album]
    artist.get_ep_singles.return_value = []
    session.artist.return_value = artist

    assert atmos.find_atmos_track_id(session, source_track) == 9001

    # Cache: a second call must not re-walk the discography.
    session.artist.reset_mock()
    assert atmos.find_atmos_track_id(session, source_track) == 9001
    session.artist.assert_not_called()


def test_match_twin_track_falls_back_to_track_number_and_title():
    source = SimpleNamespace(
        title="Harder, Better", isrc="", track_num=4, volume_num=1
    )
    wrong = SimpleNamespace(
        id=1, title="Harder, Better", isrc="", track_num=9, volume_num=1
    )
    right = SimpleNamespace(
        id=2, title="Harder, Better", isrc="", track_num=4, volume_num=1
    )
    assert atmos._match_twin_track(source, [wrong, right]) is right


# ---------------------------------------------------------------------------
# Player integration
# ---------------------------------------------------------------------------


def _bare_player() -> "object":
    from app.audio.player import PCMPlayer

    player = PCMPlayer.__new__(PCMPlayer)
    player._manifest_cache = MagicMock()
    player._manifest_cache.lookup.return_value = None
    player._local_lookup = None
    player._quality_clamp = None
    return player


def test_resolve_source_falls_back_to_lossless_when_atmos_unavailable():
    player = _bare_player()
    player._session_getter = lambda: MagicMock()
    seen = []

    def fake_uncached(
        session, track_id, quality, cache_key, t0, *, set_current_meta=True
    ):
        seen.append((track_id, quality, cache_key))
        if quality == "atmos":
            raise atmos.AtmosUnavailable("no Atmos mix")
        return (["http://seg/0"], 120.0, "info", {}, None)

    player._resolve_uncached = fake_uncached

    urls, duration, info, _, _ = player._resolve_source("999", "atmos")

    assert seen == [
        ("999", "atmos", ("999", "atmos")),
        ("999", "hi_res_lossless", ("999", "hi_res_lossless")),
    ]
    assert urls == ["http://seg/0"]
    assert duration == 120.0


def _auth_error(status: int = 401) -> Exception:
    exc = RuntimeError(f"HTTP {status}")
    exc.response = SimpleNamespace(status_code=status)
    return exc


def test_atmos_fallback_auth_error_refreshes_and_retries_original_quality():
    """A 401 on the lossless fallback must refresh, not fail the play.

    The fallback runs inside the AtmosUnavailable handler, so its
    exception used to skip the auth-retry clause entirely. It now
    re-resolves at the *original* quality: the refreshed session may
    well be entitled to Atmos.
    """
    from app.audio.player import PCMPlayer

    player = PCMPlayer.__new__(PCMPlayer)
    player._manifest_cache = ManifestCache(ttl_seconds=60)
    player._local_lookup = None
    player._quality_clamp = None
    player._session_getter = lambda: MagicMock()
    player._force_refresh = MagicMock(return_value=True)
    seen = []

    def fake_uncached(
        session, track_id, quality, cache_key, t0, *, set_current_meta=True
    ):
        seen.append(quality)
        if quality == "atmos" and len(seen) == 1:
            raise atmos.AtmosUnavailable("no Atmos mix")
        if quality == "hi_res_lossless":
            raise _auth_error(401)
        return (["http://seg/0"], 120.0, "info", {}, None)

    player._resolve_uncached = fake_uncached

    urls, duration, _info, _, _ = player._resolve_source("999", "atmos")

    assert seen == ["atmos", "hi_res_lossless", "atmos"]
    player._force_refresh.assert_called_once()
    assert urls == ["http://seg/0"]
    assert duration == 120.0


def test_asset_not_ready_error_is_not_retried_after_refresh():
    """subStatus 4005 is a per-track refusal; refreshing cannot fix it."""
    from app import tidal_client
    from app.audio.player import PCMPlayer

    player = PCMPlayer.__new__(PCMPlayer)
    player._manifest_cache = ManifestCache(ttl_seconds=60)
    player._local_lookup = None
    player._quality_clamp = None
    player._session_getter = lambda: MagicMock()
    player._force_refresh = MagicMock(return_value=True)

    def fake_uncached(
        session, track_id, quality, cache_key, t0, *, set_current_meta=True
    ):
        raise tidal_client.TidalAssetUnavailable(
            "TIDAL isn't serving this track for playback (asset not ready)."
        )

    player._resolve_uncached = fake_uncached

    with pytest.raises(tidal_client.TidalAssetUnavailable):
        player._resolve_source("999", "high_lossless")

    player._force_refresh.assert_not_called()


def test_resolve_atmos_builds_dolby_atmos_stream_info():
    player = _bare_player()
    player._manifest_cache = ManifestCache(ttl_seconds=60)
    player._build_track_meta = lambda track, duration: {"title": "x"}
    player._current_track_meta = None

    track = SimpleNamespace(duration=245)
    stream = atmos.AtmosStream(
        urls=["http://cdn/init", "http://cdn/1"],
        sample_rate=48000,
        channels=6,
        track_replay_gain=-7.2,
        track_peak=0.97,
    )

    with patch.object(
        atmos, "resolve_atmos_playback", return_value=(track, stream)
    ) as resolver:
        urls, duration, info, _, meta = player._resolve_atmos(
            MagicMock(), "42", ("42", "atmos"), time.monotonic()
        )

    resolver.assert_called_once()
    assert urls == ["http://cdn/init", "http://cdn/1"]
    assert duration == 245.0
    assert info.codec == "eac3"
    assert info.audio_mode == "DOLBY_ATMOS"
    assert info.audio_quality == "DOLBY_ATMOS"
    assert info.sample_rate_hz == 48000
    assert info.track_replay_gain_db == pytest.approx(-7.2)
    assert meta == {"title": "x"}


def test_resolve_atmos_without_track_metadata_keeps_previous_meta():
    player = _bare_player()
    player._manifest_cache = ManifestCache(ttl_seconds=60)
    player._build_track_meta = MagicMock()
    player._current_track_meta = {"title": "previous"}

    stream = atmos.AtmosStream(
        urls=["http://cdn/init", "http://cdn/1"],
        sample_rate=48000,
        channels=6,
    )

    with patch.object(
        atmos, "resolve_atmos_playback", return_value=(None, stream)
    ):
        urls, duration, info, _, meta = player._resolve_atmos(
            MagicMock(), "42", ("42", "atmos"), time.monotonic()
        )

    assert urls == ["http://cdn/init", "http://cdn/1"]
    assert duration is None
    assert meta is None
    assert info.audio_mode == "DOLBY_ATMOS"
    assert player._current_track_meta == {"title": "previous"}
    player._build_track_meta.assert_not_called()


def test_resolve_atmos_playback_tolerates_missing_v1_metadata():
    """Metadata must not veto a manifest the service will serve.

    Ids that live only in the v2 catalog 404 on `tracks/{id}` yet
    stream fine; the resolver now skips the twin lookup and asks the
    manifest service for the requested id directly.
    """
    session = _FakeSession([_manifest_payload()])
    session.track = MagicMock(side_effect=_HttpError(404))

    track, stream = atmos.resolve_atmos_playback(session, 251363269)

    assert track is None
    assert stream.urls == [
        "https://cdn.example.test/audio/ec3-5_1/init.mp4",
        "https://cdn.example.test/audio/ec3-5_1/seg-00001.mp4",
        "https://cdn.example.test/audio/ec3-5_1/seg-00002.mp4",
        "https://cdn.example.test/audio/ec3-5_1/seg-00003.mp4",
    ]
    assert session.request.calls[0]["path"] == "trackManifests/251363269"


def test_resolve_atmos_playback_keeps_metadata_when_available():
    session = _FakeSession([_manifest_payload()])
    track = SimpleNamespace(id=42, audio_modes=["DOLBY_ATMOS"])
    session.track = MagicMock(return_value=track)

    got_track, stream = atmos.resolve_atmos_playback(session, 42)

    assert got_track is track
    assert stream.urls


def test_is_atmos_stream_info_matches_backend_stream_info():
    from app.audio.player import StreamInfo, _is_atmos_stream_info

    assert _is_atmos_stream_info(StreamInfo(source="stream", audio_mode="DOLBY_ATMOS"))
    assert not _is_atmos_stream_info(StreamInfo(source="stream", audio_mode="STEREO"))
    assert not _is_atmos_stream_info(None)


def test_dolby_playback_kind_requires_objects_and_a_renderer():
    from app.audio.player import StreamInfo, _dolby_playback_kind

    atmos = StreamInfo(source="local", codec="eac3", sample_rate_hz=48000)
    assert (
        _dolby_playback_kind(
            atmos, native=True, spatial_format="atmos-headphones"
        )
        == "atmos"
    )
    # Spatial Sound off, or a virtualiser that only moves the bed.
    assert _dolby_playback_kind(atmos, native=True, spatial_format="") == "dolby_audio"
    assert (
        _dolby_playback_kind(atmos, native=True, spatial_format="windows-sonic")
        == "dolby_audio"
    )
    # The PCM engine drops the JOC objects before Windows ever sees them,
    # whatever the endpoint would have done with them.
    assert (
        _dolby_playback_kind(
            atmos, native=False, spatial_format="atmos-headphones"
        )
        == "dolby_audio"
    )


def test_dolby_playback_kind_needs_dolbys_decoder():
    from app.audio.player import StreamInfo, _dolby_playback_kind

    atmos = StreamInfo(source="local", codec="eac3")
    # Empty module list = the MFT hasn't loaded yet, not "Windows
    # downmixed"; a populated list with no Dolby module is the proof.
    assert (
        _dolby_playback_kind(
            atmos, native=True, spatial_format="atmos-headphones", decoders=[]
        )
        == "atmos"
    )
    assert (
        _dolby_playback_kind(
            atmos,
            native=True,
            spatial_format="atmos-headphones",
            decoders=["ac3dec.dll"],
        )
        == "dolby_audio"
    )
    assert (
        _dolby_playback_kind(
            atmos,
            native=True,
            spatial_format="atmos-headphones",
            decoders=["ac3dec.dll", "DolbyDDPDecMft.dll"],
        )
        == "atmos"
    )


def test_dolby_playback_kind_ignores_non_dolby_content():
    from app.audio.player import StreamInfo, _dolby_playback_kind

    # The local probe marks every >2-channel file DOLBY_ATMOS, so a 5.1
    # FLAC must not be badged as Dolby.
    surround_flac = StreamInfo(
        source="local", codec="flac", audio_mode="DOLBY_ATMOS"
    )
    assert (
        _dolby_playback_kind(
            surround_flac, native=False, spatial_format="atmos-headphones"
        )
        is None
    )
    assert _dolby_playback_kind(None, native=True, spatial_format="") is None
    for codec in ("ac3", "ac4", "truehd"):
        assert (
            _dolby_playback_kind(
                StreamInfo(source="local", codec=codec),
                native=False,
                spatial_format="",
            )
            == "dolby_audio"
        )


def test_snapshot_payload_carries_immersive_with_stream_info():
    from app.audio.player import PlayerSnapshot, StreamInfo
    import server

    snap = PlayerSnapshot(
        state="playing",
        track_id="42",
        position_ms=1000,
        duration_ms=2000,
        volume=100,
        muted=False,
        stream_info=StreamInfo(
            source="local", codec="eac3", sample_rate_hz=48000
        ),
        immersive="atmos",
    )

    payload = server._snapshot_dict(snap)

    assert payload["stream_info"]["immersive"] == "atmos"
    # Nothing loaded means no stream_info, hence no badge.
    idle = PlayerSnapshot(
        state="idle",
        track_id=None,
        position_ms=0,
        duration_ms=0,
        volume=100,
        muted=False,
    )
    assert server._snapshot_dict(idle)["stream_info"] is None

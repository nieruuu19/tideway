"""FLAC (High / Max) resolution through the OpenAPI manifest route.

The v1 playback-info endpoint answers AAC 320 for a LOSSLESS or
HI_RES_LOSSLESS request on some OAuth clients — which is how an album
tagged "Lossless" ended up playing the Low badge. `trackManifests`
still serves the real FLAC asset to the same session, so
`fetch_lossless_stream` reads that DASH manifest instead. These tests
pin the tier pick, the representation filter, and the fact that every
failure is a fallback signal (the caller keeps its AAC stream) rather
than an error.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from app import lossless


FLAC_MPD = """<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" mediaPresentationDuration="PT10S">
  <BaseURL>https://cdn.example.test/flac/</BaseURL>
  <Period duration="PT10S">
    <AdaptationSet contentType="audio" mimeType="audio/mp4">
      <SegmentTemplate timescale="1000"
                       initialization="$RepresentationID$/init.mp4"
                       media="$RepresentationID$/seg-$Number%05d$.mp4">
        <SegmentTimeline>
          <S t="0" d="5000" r="1"/>
        </SegmentTimeline>
      </SegmentTemplate>
      <Representation id="cd" codecs="flac" bandwidth="1411200"
                      audioSamplingRate="44100" audioBitDepth="16">
        <AudioChannelConfiguration
            schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011"
            value="2"/>
      </Representation>
      <Representation id="hires" codecs="flac" bandwidth="4608000"
                      audioSamplingRate="96000" audioBitDepth="24">
        <AudioChannelConfiguration
            schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011"
            value="2"/>
      </Representation>
      <Representation id="aac" codecs="mp4a.40.2" bandwidth="256000"
                      audioSamplingRate="44100">
        <AudioChannelConfiguration
            schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011"
            value="2"/>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
"""


def _hires_only_mpd() -> str:
    return FLAC_MPD.replace(
        """      <Representation id="cd" codecs="flac" bandwidth="1411200"
                      audioSamplingRate="44100" audioBitDepth="16">
        <AudioChannelConfiguration
            schemeIdUri="urn:mpeg:dash:23003:3:audio_channel_configuration:2011"
            value="2"/>
      </Representation>
""",
        "",
    )


def _aac_only_mpd() -> str:
    return FLAC_MPD.replace('codecs="flac"', 'codecs="mp4a.40.2"')


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


def _payload(xml: str = FLAC_MPD, **attrs) -> dict:
    attributes = {
        "formats": ["FLAC"],
        "uri": _data_uri(xml),
        "trackPresentation": "FULL",
        "hash": "flac-hash",
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


def _fetch(session, *, hi_res: bool = False):
    return lossless.fetch_lossless_stream(session, 4242, hi_res=hi_res)


def test_high_takes_the_cd_rate_flac_representation():
    session = _FakeSession([_payload()])

    stream = _fetch(session)

    assert stream.urls == [
        "https://cdn.example.test/flac/cd/init.mp4",
        "https://cdn.example.test/flac/cd/seg-00001.mp4",
        "https://cdn.example.test/flac/cd/seg-00002.mp4",
    ]
    assert stream.codec == "flac"
    assert stream.bit_depth == 16
    assert stream.sample_rate == 44100
    assert stream.channels == 2
    assert stream.is_hi_res is False
    assert stream.tier == "high_lossless"
    # The normalization facts follow the manifest, so the same leveling
    # the Atmos pipeline applies works here.
    assert stream.track_replay_gain == -6.5
    assert stream.track_peak == 0.98
    assert stream.album_replay_gain == -5.0
    assert stream.album_peak == 0.95
    # The AAC representation in the same adaptation set is ignored.
    assert all("/aac/" not in url for url in stream.urls)

    call = session.request.calls[0]
    assert call["path"] == "trackManifests/4242"
    assert call["params"]["formats"] == "FLAC"
    assert call["params"]["manifestType"] == "MPEG_DASH"
    assert call["params"]["usage"] == "PLAYBACK"


def test_max_takes_the_best_flac_representation():
    session = _FakeSession([_payload()])

    stream = _fetch(session, hi_res=True)

    assert stream.urls[0] == "https://cdn.example.test/flac/hires/init.mp4"
    assert stream.bit_depth == 24
    assert stream.sample_rate == 96000
    assert stream.is_hi_res is True
    assert stream.tier == "hi_res_lossless"


def test_high_falls_back_to_hi_res_when_that_is_all_there_is():
    """A request for CD-rate FLAC must not give up on a hi-res-only
    master: it is still lossless, and the reported tier says so."""
    session = _FakeSession([_payload(_hires_only_mpd())])

    stream = _fetch(session)

    assert stream.urls[0] == "https://cdn.example.test/flac/hires/init.mp4"
    assert stream.tier == "hi_res_lossless"


def test_a_track_with_only_aac_is_unavailable():
    session = _FakeSession([_payload(_aac_only_mpd()), _payload(_aac_only_mpd())])

    with pytest.raises(lossless.LosslessUnavailable):
        _fetch(session)


def test_a_manifest_without_the_flac_format_is_unavailable():
    session = _FakeSession(
        [_payload(formats=["AACLC"]), _payload(formats=["AACLC"])]
    )

    with pytest.raises(lossless.LosslessUnavailable, match="FLAC manifest"):
        _fetch(session)


def test_playback_usage_falls_back_to_download():
    from app import atmos

    session = _FakeSession([atmos.AtmosUnavailable("nope"), _payload()])

    stream = _fetch(session)

    assert stream.codec == "flac"
    assert [c["params"]["usage"] for c in session.request.calls] == [
        "PLAYBACK",
        "DOWNLOAD",
    ]


def test_a_drm_manifest_is_unavailable():
    xml = FLAC_MPD.replace(
        '<SegmentTemplate timescale="1000"',
        '<ContentProtection schemeIdUri="urn:mpeg:dash:mp4protection:2011"/>'
        '<SegmentTemplate timescale="1000"',
    )
    session = _FakeSession([_payload(xml), _payload(xml)])

    with pytest.raises(lossless.LosslessUnavailable, match="DRM"):
        _fetch(session)


def test_both_usages_unavailable_reports_the_last_error():
    from app import atmos

    session = _FakeSession(
        [atmos.AtmosUnavailable("playback refused"), atmos.AtmosUnavailable("download refused")]
    )

    with pytest.raises(lossless.LosslessUnavailable, match="download refused"):
        _fetch(session)

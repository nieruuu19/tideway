"""TIDAL's ReplayGain tags are normalized to a different reference.

TIDAL ships `trackReplayGain` aiming at about -14 LUFS, while every
leveled path in Tideway (locally measured downloads, Atmos) targets
-16 like Apple's Sound Check. Streaming a track with only its tag
therefore played ~2 dB louder than the same song played from disk.
`_tag_reference_shift_db` closes that gap for stream tags while leaving
locally tagged files — whose tagger follows the ReplayGain spec — alone.

Measured reference (downloaded "As It Was"): tag -7.76 dB for a master
our R128 meter puts at -5.86 LUFS, and Apple's Sound Check cut -10.2 dB
on the same song.
"""
from __future__ import annotations

import pytest

from app.audio.player import StreamInfo, _tag_reference_shift_db
from app.audio.replaygain import (
    TIDAL_TAG_REFERENCE_LUFS,
    ReplayGainTags,
    compute_gain_db,
)

AS_IT_WAS = ReplayGainTags(
    track_gain_db=-7.76, track_peak=1.0325, album_gain_db=0.0, album_peak=1.0
)


def _stream(**tags) -> StreamInfo:
    return StreamInfo(source="stream", **tags)


def _local(**tags) -> StreamInfo:
    return StreamInfo(source="local", **tags)


def test_shift_is_the_gap_between_tidal_and_our_reference():
    assert TIDAL_TAG_REFERENCE_LUFS == pytest.approx(-14.0)
    assert _tag_reference_shift_db(_stream()) == pytest.approx(-2.0)


def test_local_tags_and_idle_player_are_not_shifted():
    assert _tag_reference_shift_db(_local()) == 0.0
    assert _tag_reference_shift_db(None) == 0.0


def test_stream_tag_lands_near_the_measured_download():
    """-7.76 dB tag + -2 dB shift = -9.76 dB, against -10.14 dB from our
    own measurement and -10.2 dB from Apple's Sound Check."""
    shifted = compute_gain_db(
        AS_IT_WAS, "track", 0.0, True, _tag_reference_shift_db(_stream())
    )
    assert shifted == pytest.approx(-9.76)
    unshifted = compute_gain_db(AS_IT_WAS, "track", 0.0, True)
    assert shifted < unshifted


def test_album_mode_zero_gain_is_shifted_too():
    """TIDAL sends album_gain_db=0.0 for albums already at its
    reference; that is data, not a missing tag, so it shifts."""
    gain = compute_gain_db(
        AS_IT_WAS, "album", 0.0, True, _tag_reference_shift_db(_stream())
    )
    assert gain == pytest.approx(-2.0)


def test_preamp_still_composes_with_the_shift():
    gain = compute_gain_db(
        AS_IT_WAS, "track", 2.5, False, _tag_reference_shift_db(_stream())
    )
    assert gain == pytest.approx(-7.76 + 2.5 - 2.0)


def test_missing_tag_stays_flat_even_with_the_shift():
    """No tag is "no data", not "play 2 dB quiet" — the measured
    fallback owns that case."""
    sparse = ReplayGainTags()
    assert (
        compute_gain_db(sparse, "track", 0.0, True, -2.0) == 0.0
    )


def test_off_mode_ignores_the_shift():
    assert compute_gain_db(AS_IT_WAS, "off", 0.0, True, -2.0) == 0.0


def test_clipping_guard_still_wins_over_the_shift():
    hot = ReplayGainTags(track_gain_db=6.0, track_peak=1.0)
    assert compute_gain_db(hot, "track", 0.0, True, -2.0) == pytest.approx(0.0)

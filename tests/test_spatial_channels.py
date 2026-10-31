"""Channel-count selection for surround sources.

Windows Spatial Sound is a shared-mode APO, and it only virtualises a
multichannel bed. Its virtualiser wants the endpoint's full layout, so
a 5.1 Atmos core has to be padded to the endpoint's 7.1 rather than
sent short. Exclusive mode bypasses the APO chain, and every backend
has to downmix when the device genuinely can't take the source.
"""
from __future__ import annotations

import pytest

from app.audio.player import _resolve_surround_channels


@pytest.mark.parametrize(
    "source,device_max,expected",
    [
        # Unknown endpoint capability: keep the source layout.
        (6, None, 6),
        (8, None, 8),
        # Stereo endpoint always gets a stereo downmix.
        (6, 2, 2),
        (8, 2, 2),
        # Endpoints that can take exactly the source layout.
        (6, 6, 6),
        (8, 6, 6),
        (8, 8, 8),
        # Stereo source isn't touched by this helper's callers.
        (2, 8, 2),
    ],
)
def test_layouts_that_do_not_depend_on_spatial(
    source, device_max, expected
):
    assert (
        _resolve_surround_channels(
            source, device_max, windows=True, exclusive=False
        )
        == expected
    )
    assert (
        _resolve_surround_channels(
            source, device_max, windows=False, exclusive=False
        )
        == expected
    )


def test_pads_51_to_71_on_windows_shared():
    """The spatial virtualiser needs the 7.1 bed; feeding it 5.1 skips
    the height/back speakers the upmix would otherwise place."""
    assert (
        _resolve_surround_channels(
            6, 8, windows=True, exclusive=False
        )
        == 8
    )


def test_exclusive_mode_never_pads():
    """Exclusive mode is bit-perfect and bypasses the spatial APO, so
    padding would just be an unnecessary upmix."""
    assert (
        _resolve_surround_channels(
            6, 8, windows=True, exclusive=True
        )
        == 6
    )


def test_non_windows_shared_mode_never_pads():
    assert (
        _resolve_surround_channels(
            6, 8, windows=False, exclusive=False
        )
        == 6
    )


def test_51_endpoint_is_not_padded_to_seven_point_one():
    """device_max 6 can't take 7.1; asking for 8 would fail the open."""
    assert (
        _resolve_surround_channels(
            6, 6, windows=True, exclusive=False
        )
        == 6
    )

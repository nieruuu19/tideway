"""Tests for name-based output-device resolution (issue #245).

`resolve_output_device` is the fix for playback breaking when the saved
PortAudio index drifts onto a different device. It maps a saved device
*name* to a live output-capable index, and reports when the saved
device is gone so the player can fall back to the system default
instead of opening an output stream on, say, a microphone.
"""
from app.audio import output_devices


# The exact device layout from the issue #245 diagnostic: the Bose is
# output-capable, index 1 is the MacBook mic (input only), and the
# built-in speakers are output-capable. A saved index of "1" used to
# resolve to the mic and crash sd.OutputStream.
_DEVICES = [
    {"name": "Bose Flex SE SoundLink", "max_output_channels": 2,
     "max_input_channels": 0, "hostapi": 0},
    {"name": "MacBook Pro Microphone", "max_output_channels": 0,
     "max_input_channels": 1, "hostapi": 0},
    {"name": "MacBook Pro Speakers", "max_output_channels": 2,
     "max_input_channels": 0, "hostapi": 0},
]


def _patch_devices(monkeypatch, devices):
    monkeypatch.setattr(
        output_devices.sd, "query_devices", lambda: devices
    )


def test_empty_name_is_system_default(monkeypatch):
    _patch_devices(monkeypatch, _DEVICES)
    monkeypatch.setattr(output_devices, "wasapi_host_api_index", lambda: None)
    assert output_devices.resolve_output_device("") == (None, True)


def _patch_wasapi_default(monkeypatch, default):
    monkeypatch.setattr(output_devices, "wasapi_host_api_index", lambda: 2)

    def query_hostapis(idx):
        assert idx == 2
        if isinstance(default, Exception):
            raise default
        return {"name": "Windows WASAPI", "default_output_device": default}

    monkeypatch.setattr(output_devices.sd, "query_hostapis", query_hostapis)


def test_system_default_on_windows_is_the_wasapi_default(monkeypatch):
    """PortAudio's own default on Windows is MME's: a deeper output
    buffer than WASAPI on the same device, and no Exclusive Mode."""
    _patch_wasapi_default(monkeypatch, 8)
    assert output_devices.resolve_output_device("") == (8, True)


def test_no_wasapi_default_keeps_portaudios_default(monkeypatch):
    _patch_wasapi_default(monkeypatch, -1)  # no active output device
    assert output_devices.resolve_output_device("") == (None, True)


def test_wasapi_query_failure_keeps_portaudios_default(monkeypatch):
    _patch_wasapi_default(monkeypatch, RuntimeError("PortAudio not initialized"))
    assert output_devices.resolve_output_device("") == (None, True)


def test_output_device_resolves_to_its_current_index(monkeypatch):
    _patch_devices(monkeypatch, _DEVICES)
    assert output_devices.resolve_output_device(
        "MacBook Pro Speakers"
    ) == (2, True)
    assert output_devices.resolve_output_device(
        "Bose Flex SE SoundLink"
    ) == (0, True)


def test_input_only_device_is_not_available(monkeypatch):
    """The #245 crash: the saved slot now points at a microphone with
    zero output channels. It must report unavailable, not hand back an
    index we'd try to open an output stream on."""
    _patch_devices(monkeypatch, _DEVICES)
    assert output_devices.resolve_output_device(
        "MacBook Pro Microphone"
    ) == (None, False)


def test_legacy_numeric_id_matches_nothing(monkeypatch):
    """A pre-migration index string won't match any device name."""
    _patch_devices(monkeypatch, _DEVICES)
    assert output_devices.resolve_output_device("1") == (None, False)


def test_unplugged_device_is_not_available(monkeypatch):
    _patch_devices(monkeypatch, _DEVICES)
    assert output_devices.resolve_output_device(
        "USB DAC that got unplugged"
    ) == (None, False)


def test_query_failure_reports_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("PortAudio not initialized")

    monkeypatch.setattr(output_devices.sd, "query_devices", _boom)
    assert output_devices.resolve_output_device(
        "MacBook Pro Speakers"
    ) == (None, False)


def test_duplicate_name_prefers_wasapi_on_windows(monkeypatch):
    """When the same name appears under several host APIs (the Windows
    duplicate-enumeration case), prefer the WASAPI copy so we land on
    the entry the picker showed."""
    devices = [
        {"name": "Speakers", "max_output_channels": 2,
         "max_input_channels": 0, "hostapi": 0},   # MME
        {"name": "Speakers", "max_output_channels": 2,
         "max_input_channels": 0, "hostapi": 3},   # WASAPI
    ]
    _patch_devices(monkeypatch, devices)
    monkeypatch.setattr(
        output_devices, "wasapi_host_api_index", lambda: 3
    )
    assert output_devices.resolve_output_device("Speakers") == (1, True)


# ---------------------------------------------------------------------------
# Surround routing: the WASAPI copy of an endpoint often reports a stereo
# mix format even when the OS spatial APO has a 5.1/7.1 bed, while the
# MME/DirectSound copies accept the real layout.
# ---------------------------------------------------------------------------

_HOSTAPIS = [
    {"name": "MME"},
    {"name": "Windows DirectSound"},
    {"name": "Windows WASAPI"},
]

_SURROUND_DEVICES = [
    {"name": "Speakers", "max_output_channels": 8,
     "max_input_channels": 0, "hostapi": 0},   # MME
    {"name": "Speakers", "max_output_channels": 8,
     "max_input_channels": 0, "hostapi": 1},   # DirectSound
    {"name": "Speakers", "max_output_channels": 2,
     "max_input_channels": 0, "hostapi": 2},   # WASAPI
    {"name": "Headphones", "max_output_channels": 8,
     "max_input_channels": 0, "hostapi": 0},   # other device
]


def _patch_surround(monkeypatch, accepting):
    monkeypatch.setattr(
        output_devices.sd, "query_devices",
        lambda *a, **kw: _SURROUND_DEVICES if not a and not kw
        else _SURROUND_DEVICES[2],
    )
    monkeypatch.setattr(
        output_devices.sd, "query_hostapis",
        lambda index: _HOSTAPIS[index],
    )
    monkeypatch.setattr(
        output_devices.sd, "check_output_settings",
        lambda **kw: None if kw["device"] in accepting
        else (_ for _ in ()).throw(RuntimeError("Invalid number of channels")),
    )


def test_surround_resolver_prefers_mme_copy(monkeypatch):
    """MME accepts the bed, WASAPI does not → the MME copy wins even
    though WASAPI is what the picker showed."""
    _patch_surround(monkeypatch, accepting={0})

    assert output_devices.resolve_surround_output_device(
        "Speakers", 6, 48000, "float32"
    ) == (0, "mme")


def test_surround_resolver_prefers_directsound_over_mme(monkeypatch):
    _patch_surround(monkeypatch, accepting={1})

    assert output_devices.resolve_surround_output_device(
        "Speakers", 6, 48000, "float32"
    ) == (1, "windows directsound")


def test_surround_resolver_returns_none_when_no_copy_accepts(monkeypatch):
    _patch_surround(monkeypatch, accepting=set())

    assert output_devices.resolve_surround_output_device(
        "Speakers", 6, 48000, "float32"
    ) == (None, None)


def test_surround_resolver_never_switches_to_another_device(monkeypatch):
    """Headphones also accept the bed, but the user picked Speakers."""
    _patch_surround(monkeypatch, accepting={0, 3})

    index, _ = output_devices.resolve_surround_output_device(
        "Speakers", 6, 48000, "float32"
    )

    assert index == 0


def test_surround_resolver_uses_default_device_name(monkeypatch):
    """With no saved selection the system default's *name* is used, so
    the fallback still lands on that same logical device."""
    _patch_surround(monkeypatch, accepting={0})

    assert output_devices.resolve_surround_output_device(
        "", 6, 48000, "float32"
    ) == (0, "mme")

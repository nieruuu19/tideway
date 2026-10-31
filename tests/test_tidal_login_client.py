"""Client selection for the device-code login.

TIDAL binds an access token to the OAuth client that minted it and picks
the playback manifest variant (DRM-wrapped or clear) from that binding,
so the selected client must be applied before login, restored from the
session file before a refresh, and never swapped under a live session.
"""
from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from app import api_keys, tidal_client


class _Config:
    client_id = ""
    client_secret = ""


class _Session:
    is_pkce = False

    def __init__(self):
        self.config = _Config()

    def login_oauth(self):
        self.seen_login = (self.config.client_id, self.config.client_secret)
        login = SimpleNamespace(
            verification_uri_complete="link.tidal.com/ABCDE",
            user_code="ABCDE",
        )
        return login, Future()


def _skeleton(*, is_pkce: bool = False, index: int | None = 5):
    client = tidal_client.TidalClient.__new__(tidal_client.TidalClient)
    client.session = _Session()
    client.session.is_pkce = is_pkce
    client._device_code_index = index
    client._login_future = None
    return client


class _Settings:
    def __init__(self, index: int):
        self.tidal_api_key_index = index


@pytest.fixture
def settings_store(monkeypatch):
    store = {"settings": _Settings(5), "saved": []}

    monkeypatch.setattr(
        tidal_client, "load_settings", lambda: store["settings"]
    )
    monkeypatch.setattr(
        tidal_client, "save_settings", lambda value: store["saved"].append(value)
    )
    return store


def test_start_oauth_login_applies_selected_client(settings_store):
    settings_store["settings"].tidal_api_key_index = 1
    client = _skeleton()

    client.start_oauth_login()

    fire_tv = api_keys.get_item(1)
    assert client.session.seen_login == (fire_tv.client_id, fire_tv.client_secret)
    assert client._device_code_index == 1


def test_select_api_key_persists_and_signs_out_device_session(settings_store):
    client = _skeleton(index=5)
    calls = []
    client.logout = lambda: calls.append("logout")

    assert client.select_api_key(4) is True

    assert [value.tidal_api_key_index for value in settings_store["saved"]] == [4]
    assert calls == ["logout"]


def test_select_api_key_is_a_noop_for_the_current_client(settings_store):
    client = _skeleton(index=5)
    calls = []
    client.logout = lambda: calls.append("logout")

    assert client.select_api_key(5) is False

    assert calls == []
    assert settings_store["saved"] == []


def test_select_api_key_keeps_a_pkce_session_running(settings_store):
    client = _skeleton(is_pkce=True, index=5)
    calls = []
    client.logout = lambda: calls.append("logout")

    assert client.select_api_key(4) is False

    assert calls == []
    assert settings_store["saved"][0].tidal_api_key_index == 4


def test_select_api_key_rejects_unknown_clients(settings_store):
    client = _skeleton()
    with pytest.raises(ValueError):
        client.select_api_key(999)


def test_restore_device_client_pins_the_stored_client(settings_store):
    client = _skeleton(index=None)

    client._restore_device_client(4)

    tv = api_keys.get_item(4)
    assert client.session.config.client_id == tv.client_id
    assert client._device_code_index == 4


def test_restore_device_client_falls_back_to_the_legacy_tv_client(settings_store):
    client = _skeleton(index=5)

    client._restore_device_client(None)

    assert (
        client.session.config.client_id
        == tidal_client._LEGACY_DEVICE_CODE_CLIENT_ID
    )
    # Saved back as "unknown" so the next write doesn't claim the token
    # belongs to the currently selected client.
    assert client._device_code_index is None

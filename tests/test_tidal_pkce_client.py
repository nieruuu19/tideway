"""Client selection for the PKCE (browser) login.

The PKCE authorization URL and the token exchange both embed the
client_id, and the refresh path reads the same pair, so the chosen
client has to be applied before the URL is generated, honoured by the
exchange, and written next to the persisted token.
"""
from __future__ import annotations

import threading

import pytest

from app import api_keys, tidal_client

MOBILE_ID = "mobile-client"
MOBILE_SECRET = "mobile-secret"


class _Config:
    def __init__(self):
        self.client_id = "device-client"
        self.client_secret = "device-secret"
        self.client_id_pkce = MOBILE_ID
        self.client_secret_pkce = MOBILE_SECRET


class _Session:
    def __init__(self):
        self.config = _Config()
        self.is_pkce = False
        self.expiry_time = None
        self.url_calls = 0
        self.exchanged_under = None

    def pkce_login_url(self):
        self.url_calls += 1
        return f"https://login.tidal.com/authorize?client_id={self.config.client_id_pkce}"

    def pkce_get_auth_token(self, redirect_url):
        self.exchanged_under = (
            self.config.client_id_pkce,
            self.config.client_secret_pkce,
        )
        return {"access_token": "a", "refresh_token": "r"}

    def process_auth_token(self, token, is_pkce_token=False):
        self.is_pkce = is_pkce_token
        self.access_token = token["access_token"]
        self.refresh_token = token["refresh_token"]
        self.token_type = "Bearer"

    def client_enable_hires(self):
        self.config.client_id = self.config.client_id_pkce
        self.config.client_secret = self.config.client_secret_pkce

    def check_login(self):
        return True


def _skeleton():
    client = tidal_client.TidalClient.__new__(tidal_client.TidalClient)
    client.session = _Session()
    client._device_code_index = 5
    client._pkce_default_client = (MOBILE_ID, MOBILE_SECRET)
    client._pkce_client_index = None
    client._pkce_url_client_index = None
    client._session_pkce_client_index = None
    client._pkce_url_cache = None
    client._sub_lock = threading.Lock()
    client.saved = []
    client.save_session = lambda: client.saved.append(True)
    return client


class _Settings:
    def __init__(self, device_index: int = 5, pkce_index: int = 0):
        self.tidal_api_key_index = device_index
        self.tidal_pkce_client_index = pkce_index


@pytest.fixture
def settings_store(monkeypatch):
    store = {"settings": _Settings(), "saved": []}

    monkeypatch.setattr(tidal_client, "load_settings", lambda: store["settings"])
    monkeypatch.setattr(
        tidal_client, "save_settings", lambda value: store["saved"].append(value)
    )
    return store


def test_pkce_url_uses_the_mobile_pair_by_default(settings_store):
    client = _skeleton()

    url = client.pkce_login_url()

    assert f"client_id={MOBILE_ID}" in url
    assert client._pkce_url_client_index == api_keys.PKCE_DEFAULT_INDEX


def test_pkce_url_uses_the_selected_client(settings_store):
    settings_store["settings"].tidal_pkce_client_index = 1
    client = _skeleton()

    url = client.pkce_login_url()

    fire_tv = api_keys.get_item(1)
    assert f"client_id={fire_tv.client_id}" in url
    assert client._pkce_url_client_index == 1


def test_changing_the_pkce_client_regenerates_the_cached_url(settings_store):
    client = _skeleton()

    first = client.pkce_login_url()
    assert client.pkce_login_url() == first
    assert client.session.url_calls == 1

    settings_store["settings"].tidal_pkce_client_index = 4
    second = client.pkce_login_url()

    assert second != first
    assert client.session.url_calls == 2
    assert api_keys.get_item(4).client_id in second


def test_exchange_uses_the_client_that_generated_the_url(settings_store):
    settings_store["settings"].tidal_pkce_client_index = 1
    client = _skeleton()
    client.pkce_login_url()

    # Selection changes between generating the URL and pasting the code.
    settings_store["settings"].tidal_pkce_client_index = 4
    ok, reason = client.complete_pkce_login(
        "https://tidal.com/android/login/auth?code=abc"
    )

    assert (ok, reason) == (True, None)
    fire_tv = api_keys.get_item(1)
    assert client.session.exchanged_under == (
        fire_tv.client_id,
        fire_tv.client_secret,
    )
    assert client._session_pkce_client_index == 1


def test_exchange_without_a_cached_url_uses_the_stored_selection(settings_store):
    settings_store["settings"].tidal_pkce_client_index = 5
    client = _skeleton()

    ok, _ = client.complete_pkce_login(
        "https://tidal.com/android/login/auth?code=abc"
    )

    assert ok
    hires = api_keys.get_item(5)
    assert client.session.exchanged_under == (
        hires.client_id,
        hires.client_secret,
    )


def test_mobile_default_stores_no_override(settings_store):
    client = _skeleton()

    client.pkce_login_url()
    client.complete_pkce_login("https://tidal.com/android/login/auth?code=abc")

    assert client._session_pkce_client_index is None
    assert client.session.config.client_id == MOBILE_ID


def test_restore_pkce_client_pins_the_stored_pair():
    client = _skeleton()

    client._restore_pkce_client(4)

    tv = api_keys.get_item(4)
    assert client.session.config.client_id_pkce == tv.client_id
    assert client.session.config.client_secret_pkce == tv.client_secret
    assert client._session_pkce_client_index == 4


def test_restore_pkce_client_falls_back_to_mobile():
    client = _skeleton()
    client.session.config.client_id_pkce = "stale"

    client._restore_pkce_client(None)

    assert client.session.config.client_id_pkce == MOBILE_ID
    assert client.session.config.client_secret_pkce == MOBILE_SECRET
    assert client._session_pkce_client_index is None


def test_select_pkce_client_persists_and_clears_the_url(settings_store):
    client = _skeleton()
    client.pkce_login_url()
    assert client._pkce_url_cache is not None

    assert client.select_pkce_api_key(1) is False

    assert [s.tidal_pkce_client_index for s in settings_store["saved"]] == [1]
    assert client._pkce_url_cache is None


def test_select_pkce_client_signs_out_a_live_pkce_session(settings_store):
    client = _skeleton()
    client.session.is_pkce = True
    calls = []
    client.logout = lambda: calls.append("logout")

    assert client.select_pkce_api_key(4) is True
    assert calls == ["logout"]


def test_select_pkce_client_rejects_unknown_indices(settings_store):
    client = _skeleton()

    with pytest.raises(ValueError):
        client.select_pkce_api_key(999)
    with pytest.raises(ValueError):
        client.select_pkce_api_key(False)

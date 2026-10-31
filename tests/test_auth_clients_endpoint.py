"""Wire-level tests for the login-client picker endpoints.

`POST /api/auth/client` has to persist the choice and tell the caller
whether the device-code session was invalidated, and neither it nor the
GET may leak client secrets to the browser.
"""
import copy

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.settings as _settings_mod
    import server

    monkeypatch.setattr(_settings_mod, "SETTINGS_FILE", tmp_path / "settings.json")
    original_settings = copy.deepcopy(server.settings)
    server.settings.offline_mode = True
    # server.settings is loaded from the developer's real settings.json
    # at import time; pin the pickers so the assertions don't depend on
    # whichever client this machine happens to have selected.
    server.settings.tidal_api_key_index = 5
    server.settings.tidal_pkce_client_index = 0
    with TestClient(server.app) as c:
        yield c
    server.settings = original_settings


def test_clients_lists_only_the_hires_client(client):
    r = client.get("/api/auth/clients")

    assert r.status_code == 200
    body = r.json()
    assert [item["index"] for item in body["clients"]] == [5]
    assert body["selected"] == 5
    for item in body["clients"]:
        assert set(item) == {"index", "platform", "formats"}


def test_clients_reports_the_hires_client_even_when_settings_are_stale(client):
    """A settings.json written by an older build can name the TV or Fire
    TV client; the payload must say what a login would really use."""
    import server

    server.settings.tidal_api_key_index = 4

    assert client.get("/api/auth/clients").json()["selected"] == 5


def test_clients_includes_the_pkce_picker_with_mobile_first(client):
    r = client.get("/api/auth/clients")

    body = r.json()
    assert [item["index"] for item in body["pkce_clients"]] == [0, 5]
    assert body["pkce_selected"] == 0
    assert body["pkce_clients"][0]["platform"] == "Tidal mobile app"
    for item in body["pkce_clients"]:
        assert set(item) == {"index", "platform", "formats"}
    assert "secret" not in r.text.lower()


def test_select_pkce_client_reports_reauth(client, monkeypatch):
    import server

    seen = []
    monkeypatch.setattr(
        server.tidal,
        "select_pkce_api_key",
        lambda index: seen.append(index) or True,
    )

    r = client.post("/api/auth/pkce-client", json={"index": 1})

    assert r.status_code == 200
    assert r.json() == {"ok": True, "index": 1, "reauth_required": True}
    assert seen == [1]
    assert server.settings.tidal_pkce_client_index == 1


def test_select_pkce_client_accepts_the_mobile_sentinel(client, monkeypatch):
    import server

    monkeypatch.setattr(
        server.tidal, "select_pkce_api_key", lambda index: False
    )

    r = client.post("/api/auth/pkce-client", json={"index": 0})

    assert r.status_code == 200
    assert r.json() == {"ok": True, "index": 0, "reauth_required": False}


def test_select_pkce_client_rejects_unknown_index(client):
    r = client.post("/api/auth/pkce-client", json={"index": 999})

    assert r.status_code == 400


def test_select_client_reports_reauth(client, monkeypatch):
    import server

    seen = []
    monkeypatch.setattr(
        server.tidal,
        "select_api_key",
        lambda index: seen.append(index) or True,
    )

    r = client.post("/api/auth/client", json={"index": 5})

    assert r.status_code == 200
    assert r.json() == {"ok": True, "index": 5, "reauth_required": True}
    assert seen == [5]
    assert server.settings.tidal_api_key_index == 5


def test_select_client_rejects_unknown_index(client):
    r = client.post("/api/auth/client", json={"index": 999})

    assert r.status_code == 400


def test_select_client_rejects_a_retired_client(client):
    """Index 4 still resolves for stored sessions but must not be
    selectable again — it cannot complete the code flow."""
    r = client.post("/api/auth/client", json={"index": 4})

    assert r.status_code == 400


def test_picker_is_available_while_signed_out(client, monkeypatch):
    """The picker renders on the login page, i.e. before there is a
    session. Gating these endpoints on `_require_local_access` made them
    401 exactly when the login screen needed them."""
    import server

    server.settings.offline_mode = False
    monkeypatch.setattr(server, "_is_logged_in", lambda: False)
    monkeypatch.setattr(server.tidal, "select_api_key", lambda index: False)
    monkeypatch.setattr(
        server.tidal, "select_pkce_api_key", lambda index: False
    )

    assert client.get("/api/auth/clients").status_code == 200
    assert (
        client.post("/api/auth/client", json={"index": 5}).status_code == 200
    )
    assert (
        client.post("/api/auth/pkce-client", json={"index": 0}).status_code
        == 200
    )

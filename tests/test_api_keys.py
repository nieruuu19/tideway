"""The selectable OAuth clients are persisted by index, so the table's
indices are part of the on-disk format: a renumbering silently switches
clients and invalidates stored logins. These tests pin the current
mapping and make sure secrets never reach the UI payload."""
from __future__ import annotations

from app import api_keys


def test_default_is_the_hires_client():
    default = api_keys.get_item(api_keys.DEFAULT_API_KEY_INDEX)
    assert default is not None
    assert default.index == 5
    assert default.platform == "Tidal HiRes"
    assert default.client_id == "fX2JxdmntZWK0ixT"


def test_index_mapping_is_stable():
    assert [key.index for key in api_keys.get_items()] == [1, 4, 5]
    assert api_keys.get_item(1).platform == "Fire TV (legacy)"
    assert api_keys.get_item(4).platform == "Tidal TV"


def test_unknown_indices_are_rejected():
    for value in (None, "5", True, 0, 2, 3, 6, 999):
        assert api_keys.get_item(value) is None


def test_public_items_omit_secrets():
    for item in api_keys.public_items():
        assert set(item) == {"index", "platform", "formats"}
    assert [item["index"] for item in api_keys.public_items()] == [1, 4, 5]


def test_every_client_has_credentials():
    for key in api_keys.get_items():
        assert key.client_id
        assert key.client_secret


def test_pkce_items_put_the_mobile_default_first():
    items = api_keys.pkce_public_items()
    assert [item["index"] for item in items] == [0, 1, 4, 5]
    assert items[0]["platform"] == "Tidal mobile app"
    assert api_keys.get_item(0) is None


def test_is_pkce_index_accepts_the_mobile_sentinel_and_table_entries():
    for value in (0, 1, 4, 5):
        assert api_keys.is_pkce_index(value)
    for value in (None, "0", True, False, 2, 999):
        assert not api_keys.is_pkce_index(value)


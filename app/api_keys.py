"""Selectable TIDAL OAuth clients for the device-code login.

Ported from tidekeeper (TIDALDL-PY) `tidal_dl/apiKey.py`. TIDAL binds
every access token to the OAuth client that minted it, and the playback
manifest service picks the delivery format from that binding: the
mobile (PKCE) client receives a Widevine-wrapped E-AC-3 stream for
Dolby Atmos, while other clients can get a clear stream the local
decoder is able to play. Changing the client therefore requires a fresh
login — see `TidalClient.select_api_key`.

Indices are persisted in settings.json and in tidal_session.json, so
entries must never be renumbered when one is retired: doing so silently
switches clients and invalidates stored logins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Index 5 is the default for the same reason tidekeeper picks it: it is
# the hi-res ("Max") client, and it supports the Limited Input Device
# flow today. The TV and legacy Fire TV clients are kept because their
# playback manifests differ from the mobile client's.
DEFAULT_API_KEY_INDEX = 5
# Sentinel for "tidalapi's built-in mobile client", the only pair known
# to be entitled for Max. Stored in settings.tidal_pkce_client_index.
PKCE_DEFAULT_INDEX = 0


@dataclass(frozen=True)
class ApiKey:
    index: int
    platform: str
    formats: str
    client_id: str
    client_secret: str


_API_KEYS: dict[int, ApiKey] = {
    1: ApiKey(
        index=1,
        platform="Fire TV (legacy)",
        formats="Legacy alternative; playback availability varies",
        client_id="7m7Ap0JC9j1cOM3n",
        client_secret="vRAdA108tlvkJpTsGZS8rGZ7xTlbJ0qaZ2K9saEzsgY=",
    ),
    4: ApiKey(
        index=4,
        platform="Tidal TV",
        formats="Normal/High/HiFi",
        client_id="4N3n6Q1x95LL5K7p",
        client_secret="oKOXfJW371cX6xaZ0PyhgGNBdNLlBZd4AKKYougMjik=",
    ),
    5: ApiKey(
        index=5,
        platform="Tidal HiRes",
        formats="Normal/High/HiFi/Max (up to 24-bit/192 kHz)",
        client_id="fX2JxdmntZWK0ixT",
        client_secret="1Nn9AfDAjxrgJFJbKNWLeAyKGVGmINuXPPLHVXAvxAg=",
    ),
}


def get_item(index: Any) -> Optional[ApiKey]:
    """Return the client for `index`, or None when it isn't selectable."""
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return _API_KEYS.get(index)


def get_items() -> list[ApiKey]:
    """Every selectable client, ordered by index."""
    return [_API_KEYS[index] for index in sorted(_API_KEYS)]


def public_items() -> list[dict[str, Any]]:
    """Client metadata safe to hand to the UI — no secrets."""
    return [
        {"index": key.index, "platform": key.platform, "formats": key.formats}
        for key in get_items()
    ]


def get_default_index() -> int:
    return DEFAULT_API_KEY_INDEX


def is_pkce_index(index: Any) -> bool:
    """True for the mobile sentinel and for table entries."""
    if isinstance(index, bool) or not isinstance(index, int):
        return False
    return index == PKCE_DEFAULT_INDEX or index in _API_KEYS


def pkce_public_items() -> list[dict[str, Any]]:
    """Clients offered by the PKCE login, mobile default first."""
    return [
        {
            "index": PKCE_DEFAULT_INDEX,
            "platform": "Tidal mobile app",
            "formats": "Default; unlocks Max (hi-res)",
        },
        *public_items(),
    ]

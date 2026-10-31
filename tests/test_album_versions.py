"""Tests for `_split_album_versions` — the "Other versions" shelf.

Tidal ships every edition of a record as its own album ID with a
variant-tagged title. The shelf on the album page is built by matching
those variants back to the record being viewed, so these cases pin the
grouping, the ordering, and the "don't show the page you're on" rule.
"""
from __future__ import annotations

from types import SimpleNamespace

from server import _normalize_album_title, _split_album_versions


def _a(aid: int | None, name: str, date: str | None = "2020-01-01"):
    return SimpleNamespace(id=aid, name=name, release_date=date)


def test_groups_editions_of_the_same_record():
    current = _a(1, "Abbey Road (2019 Mix)", "2019-09-27")
    albums = [
        _a(2, "Abbey Road (Super Deluxe Edition)", "2019-09-27"),
        _a(3, "Abbey Road - 2009 Remaster", "2009-09-09"),
        _a(4, "Let It Be", "1970-05-08"),
    ]

    versions, others = _split_album_versions(current, albums)

    # Oldest edition first, so the standard/remaster leads.
    assert [a.id for a in versions] == [3, 2]
    assert [a.id for a in others] == [4]


def test_never_lists_the_album_being_viewed():
    current = _a(1, "Abbey Road", "1969-09-26")
    albums = [_a(1, "Abbey Road", "1969-09-26")]

    versions, others = _split_album_versions(current, albums)

    assert versions == [] and others == []


def test_dedupes_repeated_ids_across_album_and_ep_lists():
    current = _a(1, "Abbey Road", "1969-09-26")
    albums = [
        _a(2, "Abbey Road (2019 Mix)", "2019-09-27"),
        _a(2, "Abbey Road (2019 Mix)", "2019-09-27"),
    ]

    versions, _ = _split_album_versions(current, albums)

    assert [a.id for a in versions] == [2]


def test_unrelated_titles_keep_one_entry_per_title():
    current = _a(1, "Abbey Road", "1969-09-26")
    albums = [
        _a(5, "Let It Be (Remastered)", "2009-09-09"),
        _a(6, "Let It Be (Deluxe)", "2021-10-15"),
        _a(7, "Help!", "1965-08-06"),
    ]

    versions, others = _split_album_versions(current, albums)

    assert versions == []
    assert [a.id for a in others] == [5, 7]


def test_undated_editions_sort_last():
    current = _a(1, "Abbey Road", "1969-09-26")
    albums = [
        _a(2, "Abbey Road (Unknown Edition)", None),
        _a(3, "Abbey Road (2019 Mix)", "2019-09-27"),
    ]

    versions, _ = _split_album_versions(current, albums)

    assert [a.id for a in versions] == [3, 2]


def test_skips_candidates_without_an_id():
    current = _a(1, "Abbey Road", "1969-09-26")
    albums = [_a(None, "Abbey Road (2019 Mix)", "2019-09-27")]

    versions, others = _split_album_versions(current, albums)

    assert versions == [] and others == []


def test_nested_tags_reduce_to_the_same_record():
    # Atmos masters arrive with the edition tag nested in a second pair
    # of parens; the scanner has to swallow the whole trailing group.
    current = _a(1, "Limasawa Street ((Dolby Atmos))", "2019-08-30")
    albums = [
        current,
        _a(2, "Limasawa Street", "2019-08-30"),
        _a(3, "Limasawa Street (Live)", "2020-01-01"),
    ]

    versions, others = _split_album_versions(current, albums)

    assert [a.id for a in versions] == [2, 3]
    assert others == []


def test_stacked_tags_are_stripped_one_at_a_time():
    current = _a(1, "Hurry Up Tomorrow (Deluxe) [Explicit]", "2025-01-31")
    albums = [_a(2, "Hurry Up Tomorrow", "2025-01-31")]

    versions, _ = _split_album_versions(current, albums)

    assert [a.id for a in versions] == [2]


def test_unbalanced_parenthetical_is_not_stripped():
    assert _normalize_album_title("Album (Live") == "album (live"
    assert _normalize_album_title("Album") == "album"


def test_trailing_tags_work_whatever_the_bracket_type():
    assert _normalize_album_title("Forever Young [2019 Remaster]") == "forever young"
    assert _normalize_album_title("Album (Deluxe) [Explicit]") == "album"
    assert _normalize_album_title("Album - Super Deluxe Edition") == "album"

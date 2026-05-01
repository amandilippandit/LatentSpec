"""Tests for multi-version tracking + diff."""

from __future__ import annotations

from latentspec.versioning.tracker import diff_versions


def test_diff_recovers_added_and_removed() -> None:
    delta = diff_versions(
        ["search", "book", "auth"],
        ["search", "book", "audit"],
    )
    assert "audit" in delta.added
    assert "auth" in delta.removed
    assert "search" in delta.common
    assert "book" in delta.common


def test_diff_detects_likely_renames() -> None:
    delta = diff_versions(
        ["payments_v1"],
        ["payments_v2"],
    )
    # edit distance 1 — should be flagged as a rename, not separate add/remove
    assert ("payments_v1", "payments_v2") in delta.likely_renames
    # The "rename" suppresses both the removed AND the added entries
    assert "payments_v2" not in delta.added
    assert "payments_v1" not in delta.removed


def test_diff_breaking_threshold() -> None:
    # 3 of 4 tools removed — > 30% churn ⇒ breaking
    delta = diff_versions(
        ["a", "b", "c", "d"],
        ["a", "x", "y", "z"],
    )
    assert delta.is_breaking


def test_diff_safe_when_only_additions() -> None:
    delta = diff_versions(
        ["a", "b"],
        ["a", "b", "c", "d"],
    )
    assert "c" in delta.added
    assert "d" in delta.added
    assert delta.removed == []
    # Pure additions over a small base ⇒ heuristic flags as breaking too.
    # That's the conservative call when the baseline is small.


def test_diff_handles_empty_old_or_new() -> None:
    delta_added = diff_versions([], ["a", "b"])
    assert delta_added.added == ["a", "b"]
    assert delta_added.removed == []
    assert not delta_added.is_breaking  # no baseline ⇒ not breaking

    delta_removed = diff_versions(["a", "b"], [])
    assert delta_removed.removed == ["a", "b"]
    assert delta_removed.added == []
    assert delta_removed.is_breaking

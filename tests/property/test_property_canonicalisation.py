"""Property-based tests for tool-name canonicalisation.

The invariants:

  - canonicalise(canonicalise(x)) == canonicalise(x)  (idempotence)
  - Two strings that have the same `canonical_form` end up in the same
    cluster (closure under the strongest equivalence).
  - The output canonical name is itself one of the cluster members.
  - apply_canonicalisation never changes the trace's structural
    fingerprint relative to canonical-form-equivalent inputs.
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck, strategies as st

from latentspec.canonicalization.canonicalizer import (
    ToolCanonicalizer,
    canonical_form,
)


tool_like = st.from_regex(r"\A[A-Za-z][A-Za-z0-9_./\-]{1,30}\Z", fullmatch=True)


@given(names=st.lists(tool_like, min_size=0, max_size=20, unique=True))
@settings(max_examples=100, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_canonicalisation_is_idempotent(names) -> None:
    """canonicalise(canonicalise(x)) == canonicalise(x) for any tool."""
    cano = ToolCanonicalizer().fit(names)
    for name in names:
        once = cano.canonicalise(name)
        twice = cano.canonicalise(once)
        # Either: `once` IS the canonical (so twice == once)
        #     or: `once` is itself in the input set and resolves to the same canonical
        assert twice == once, f"non-idempotent: {name} -> {once} -> {twice}"


@given(names=st.lists(tool_like, min_size=2, max_size=12, unique=True))
@settings(max_examples=100, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_same_canonical_form_implies_same_cluster(names) -> None:
    """If two raw names have identical `canonical_form`, they must be in
    the same alias cluster after fitting."""
    cano = ToolCanonicalizer().fit(names)
    by_form: dict[str, list[str]] = {}
    for name in names:
        by_form.setdefault(canonical_form(name), []).append(name)

    for cf, group in by_form.items():
        if len(group) < 2:
            continue
        canonicals = {cano.canonical_for[name] for name in group}
        assert len(canonicals) == 1, (
            f"names with identical canonical_form '{cf}' ended up in different clusters: "
            f"{group} -> {canonicals}"
        )


@given(names=st.lists(tool_like, min_size=1, max_size=12, unique=True))
@settings(max_examples=100, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_canonical_member_is_in_cluster(names) -> None:
    """The canonical name picked for each cluster must be one of the cluster members."""
    cano = ToolCanonicalizer().fit(names)
    for canonical, members in cano.clusters.items():
        assert canonical in members, (
            f"canonical {canonical!r} not among cluster members {members}"
        )


@given(names=st.lists(tool_like, min_size=1, max_size=12, unique=True))
@settings(max_examples=50, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_every_input_appears_in_decisions(names) -> None:
    cano = ToolCanonicalizer().fit(names)
    decided = {d.raw_name for d in cano.decisions}
    assert decided == set(names)

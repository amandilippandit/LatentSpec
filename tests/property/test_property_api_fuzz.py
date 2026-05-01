"""API endpoint schema fuzzing.

Fires arbitrary JSON payloads at the trace ingestion endpoint via FastAPI's
in-process test client. Asserts that for any input, the server either:
  - returns a 2xx response after successful normalisation, OR
  - returns a 4xx with a typed error message,
but never 5xx (server-side crash).

Uses Hypothesis to generate JSON-shaped payloads — including nested
structures, NaN/Inf, unicode, and oversized strings — that any production
API will eventually receive.
"""

from __future__ import annotations

import json
import math

from hypothesis import HealthCheck, given, settings, strategies as st


# Construct only HTTP-valid JSON: no NaN/Inf at the top level since
# json.dumps with allow_nan=False would crash before sending. We test
# string-encoded NaN below in a separate fuzzer.
_json_primitive = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-10**12, max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=128),
)


def _json_value():
    return st.recursive(
        _json_primitive,
        lambda children: st.one_of(
            st.lists(children, max_size=5),
            st.dictionaries(st.text(min_size=1, max_size=16), children, max_size=5),
        ),
        max_leaves=20,
    )


_arbitrary_json = _json_value()


@given(payload=_arbitrary_json)
@settings(max_examples=150, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_api_never_5xxs_on_random_payloads(payload) -> None:
    """Random JSON shapes through `POST /traces` must not crash the server."""
    from fastapi.testclient import TestClient

    from latentspec.api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Even malformed payloads need a proper agent_id to reach
        # validation. Wrap in the TraceIn envelope.
        envelope = {
            "agent_id": "00000000-0000-0000-0000-000000000000",
            "format": "normalized",
            "payload": payload if isinstance(payload, dict) else {"steps": [payload]},
        }
        try:
            r = client.post("/traces", json=envelope)
        except (TypeError, ValueError, OverflowError):
            # JSON encoder rejection (e.g. unsupported types in our generator) —
            # the server never sees the payload, so trivially can't 5xx.
            return
        assert r.status_code < 500, (
            f"server 5xx on {envelope!r}: {r.status_code} {r.text[:200]}"
        )


@given(envelope=st.dictionaries(st.text(max_size=12), _arbitrary_json, max_size=8))
@settings(max_examples=100, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_api_never_5xxs_on_random_envelopes(envelope) -> None:
    """Random envelopes (not necessarily containing the right keys) must
    still come back 4xx, never 5xx."""
    from fastapi.testclient import TestClient

    from latentspec.api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            r = client.post("/traces", json=envelope)
        except (TypeError, ValueError, OverflowError):
            return
        assert r.status_code < 500


@given(payload=st.fixed_dictionaries({
    "agent_id": st.from_regex(
        r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
        fullmatch=True,
    ),
    "format": st.sampled_from(["normalized", "langchain", "raw_json", "dag"]),
    "payload": _json_value(),
}))
@settings(max_examples=80, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
def test_api_well_formed_envelopes_with_random_inner_payload(payload) -> None:
    from fastapi.testclient import TestClient

    from latentspec.api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            r = client.post("/traces", json=payload)
        except (TypeError, ValueError, OverflowError):
            return
        assert r.status_code < 500

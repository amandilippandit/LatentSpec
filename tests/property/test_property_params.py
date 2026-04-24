"""Property-based tests for the per-type params schema.

The invariants:

  - validate_params either succeeds or raises ParamsValidationError;
    no other exception kind escapes.
  - When it succeeds, the returned dict round-trips through validate_params
    again (idempotence).
  - Validated dicts contain ALL the required-by-schema keys for their type.
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck, strategies as st

from latentspec.models.invariant import InvariantType
from latentspec.schemas.params import (
    ParamsValidationError,
    schema_for,
    validate_params,
)
from tests.property.strategies import PARAMS_FOR


VALIDATED_TYPES = list(PARAMS_FOR)


@given(t=st.sampled_from(VALIDATED_TYPES), data=st.data())
@settings(max_examples=200, deadline=1000, suppress_health_check=[HealthCheck.too_slow])
def test_valid_params_validate_and_round_trip(t, data) -> None:
    raw = data.draw(PARAMS_FOR[t])
    validated = validate_params(t, raw)
    # Round-trip must be a fixed point
    again = validate_params(t, validated)
    assert again == validated


@given(t=st.sampled_from(VALIDATED_TYPES), garbage=st.dictionaries(
    st.text(min_size=1, max_size=16),
    st.one_of(
        st.text(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none(),
    ),
    min_size=0,
    max_size=10,
))
@settings(max_examples=200, deadline=1000, suppress_health_check=[HealthCheck.too_slow])
def test_garbage_params_either_validate_or_raise_typed(t, garbage) -> None:
    """No `KeyError`, `AttributeError`, etc. — only ParamsValidationError."""
    try:
        validate_params(t, garbage)
    except ParamsValidationError:
        pass
    except Exception as e:
        raise AssertionError(
            f"unexpected {type(e).__name__} from validate_params({t.value}, {garbage!r}): {e}"
        ) from e


@given(t=st.sampled_from(VALIDATED_TYPES))
def test_empty_params_always_rejected_for_strict_types(t) -> None:
    if t == InvariantType.OUTPUT_FORMAT:
        return  # OutputFormatParams has all-optional fields
    try:
        validate_params(t, {})
    except ParamsValidationError:
        return
    raise AssertionError(f"empty params should not validate for {t.value}")

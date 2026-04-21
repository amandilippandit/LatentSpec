"""Hypothesis strategies for generating arbitrary §3.2 traces and params.

Used by every property test in this directory. The strategies match the
public schema constraints exactly so anything generated should validate;
anything that fails validation surfaces an inconsistency between
generator and schema (a real bug).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import strategies as st

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.params import schema_for
from latentspec.schemas.trace import (
    AgentResponseStep,
    AgentThoughtStep,
    NormalizedTrace,
    StepType,
    SystemStep,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


# ---- atomic strategies ---------------------------------------------------


# Tool names that match the schema's _TOOL_NAME_RE
tool_name = st.from_regex(r"\A[A-Za-z_][A-Za-z0-9_./\-]{0,31}\Z", fullmatch=True)

# Lower-case alphanumeric_underscore keywords
keyword = st.from_regex(r"\A[a-z][a-z0-9_]{1,31}\Z", fullmatch=True)

# Segment label
segment = st.from_regex(r"\A[A-Za-z0-9_\-]{1,16}\Z", fullmatch=True)


# Reasonable args structure — JSON-serialisable, bounded depth/breadth.
def args_value(max_leaves: int = 10) -> st.SearchStrategy:
    primitives = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-10**9, max_value=10**9),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        st.text(max_size=64),
    )
    return st.recursive(
        primitives,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(st.text(min_size=1, max_size=16), children, max_size=4),
        ),
        max_leaves=max_leaves,
    )


# ---- step strategies -----------------------------------------------------


user_input_step = st.builds(UserInputStep, content=st.text(min_size=0, max_size=256))


tool_call_step = st.builds(
    ToolCallStep,
    tool=tool_name,
    args=st.dictionaries(st.text(min_size=1, max_size=16), args_value(), max_size=4),
    latency_ms=st.one_of(st.none(), st.integers(min_value=0, max_value=60_000)),
    result_status=st.one_of(
        st.none(),
        st.sampled_from(["success", "error", "timeout", "partial"]),
    ),
    result=st.one_of(st.none(), args_value()),
)


agent_response_step = st.builds(
    AgentResponseStep, content=st.text(min_size=0, max_size=512)
)


agent_thought_step = st.builds(
    AgentThoughtStep, content=st.text(min_size=0, max_size=256)
)


system_step = st.builds(SystemStep, content=st.text(min_size=0, max_size=128))


step = st.one_of(
    user_input_step,
    tool_call_step,
    agent_response_step,
    agent_thought_step,
    system_step,
)


# ---- traces --------------------------------------------------------------


trace_metadata = st.builds(
    TraceMetadata,
    model=st.one_of(st.none(), st.text(max_size=32)),
    version=st.one_of(st.none(), st.text(max_size=32)),
    user_segment=st.one_of(st.none(), segment),
    locale=st.one_of(st.none(), st.text(max_size=8)),
)


@st.composite
def normalized_trace(
    draw,
    *,
    min_steps: int = 0,
    max_steps: int = 24,
) -> NormalizedTrace:
    """Generate an arbitrary §3.2 NormalizedTrace."""
    n_steps = draw(st.integers(min_value=min_steps, max_value=max_steps))
    steps = draw(st.lists(step, min_size=n_steps, max_size=n_steps))
    base_dt = draw(
        st.datetimes(
            min_value=datetime(2025, 1, 1),
            max_value=datetime(2027, 1, 1),
        )
    )
    base = base_dt.replace(tzinfo=UTC)
    return NormalizedTrace(
        trace_id=draw(st.text(min_size=1, max_size=64)),
        agent_id=draw(st.text(min_size=1, max_size=64)),
        timestamp=base,
        ended_at=base + timedelta(milliseconds=draw(st.integers(0, 60_000))),
        steps=steps,
        metadata=draw(trace_metadata),
    )


# ---- params strategies (one per InvariantType) ---------------------------


def ordering_params() -> st.SearchStrategy:
    return st.fixed_dictionaries({"tool_a": tool_name, "tool_b": tool_name}).filter(
        lambda d: d["tool_a"] != d["tool_b"]
    )


def conditional_params() -> st.SearchStrategy:
    return st.fixed_dictionaries({"keyword": keyword, "tool": tool_name})


def negative_params() -> st.SearchStrategy:
    """Either forbidden_patterns OR allowed_repertoire — never both."""
    return st.one_of(
        st.fixed_dictionaries(
            {"forbidden_patterns": st.lists(tool_name, min_size=1, max_size=4)}
        ),
        st.fixed_dictionaries(
            {"allowed_repertoire": st.lists(tool_name, min_size=1, max_size=8)}
        ),
    )


def statistical_params() -> st.SearchStrategy:
    latency = st.fixed_dictionaries(
        {
            "metric": st.just("latency_ms"),
            "tool": tool_name,
            "threshold": st.floats(min_value=0.0, max_value=60_000, allow_nan=False),
        }
    )
    success = st.fixed_dictionaries(
        {
            "metric": st.just("success_rate"),
            "tool": tool_name,
            "rate": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        }
    )
    envelope = st.builds(
        lambda f, p1, span: {
            "metric": "feature_envelope",
            "feature": f,
            "p1": p1,
            "p99": p1 + abs(span),
        },
        st.text(min_size=1, max_size=32),
        st.floats(min_value=0.0, max_value=1000, allow_nan=False),
        st.floats(min_value=0.1, max_value=1000, allow_nan=False),
    )
    return st.one_of(latency, success, envelope)


def state_params() -> st.SearchStrategy:
    return st.fixed_dictionaries(
        {
            "terminator_tool": tool_name,
            "forbidden_after": st.lists(tool_name, min_size=1, max_size=4),
        }
    )


def composition_params() -> st.SearchStrategy:
    return st.fixed_dictionaries(
        {"upstream_tool": tool_name, "downstream_tool": tool_name}
    ).filter(lambda d: d["upstream_tool"] != d["downstream_tool"])


def tool_selection_params() -> st.SearchStrategy:
    return st.fixed_dictionaries(
        {
            "segment": segment,
            "expected_tool": tool_name,
            "forbidden_tool": tool_name,
        }
    ).filter(lambda d: d["expected_tool"] != d["forbidden_tool"])


PARAMS_FOR = {
    InvariantType.ORDERING: ordering_params(),
    InvariantType.CONDITIONAL: conditional_params(),
    InvariantType.NEGATIVE: negative_params(),
    InvariantType.STATISTICAL: statistical_params(),
    InvariantType.STATE: state_params(),
    InvariantType.COMPOSITION: composition_params(),
    InvariantType.TOOL_SELECTION: tool_selection_params(),
}


@st.composite
def invariant_spec(draw, *, type_: InvariantType | None = None):
    from latentspec.checking.base import InvariantSpec

    chosen_type = type_ or draw(
        st.sampled_from(
            [
                InvariantType.ORDERING,
                InvariantType.CONDITIONAL,
                InvariantType.NEGATIVE,
                InvariantType.STATISTICAL,
                InvariantType.STATE,
                InvariantType.COMPOSITION,
                InvariantType.TOOL_SELECTION,
            ]
        )
    )
    params = draw(PARAMS_FOR[chosen_type])
    severity = draw(st.sampled_from(list(Severity)))
    return InvariantSpec(
        id=f"inv-{draw(st.integers(0, 10_000_000))}",
        type=chosen_type,
        description=draw(st.text(min_size=1, max_size=120)),
        formal_rule="...",
        severity=severity,
        params=params,
    )

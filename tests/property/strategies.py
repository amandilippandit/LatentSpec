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

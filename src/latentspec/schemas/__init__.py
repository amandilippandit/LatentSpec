"""Pydantic schemas implementing the §3.2 wire format and API contracts.

The §3.2 normalized JSON is the canonical exchange format between every
ingestion path (SDK, OTel, raw API) and the mining pipeline. All §5.2
framework integrations translate *into* this schema.
"""

from latentspec.schemas.invariant import (
    InvariantCandidate,
    InvariantOut,
    MinedInvariant,
)
from latentspec.schemas.params import (
    CompositionParams,
    ConditionalParams,
    NegativeParams,
    OrderingParams,
    OutputFormatParams,
    ParamsValidationError,
    StateParams,
    StatisticalParams,
    ToolSelectionParams,
    schema_for,
    validate_params,
)
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    StepType,
    ToolCallStep,
    TraceIn,
    TraceMetadata,
    TraceStep,
    UserInputStep,
)

__all__ = [
    "AgentResponseStep",
    "CompositionParams",
    "ConditionalParams",
    "InvariantCandidate",
    "InvariantOut",
    "MinedInvariant",
    "NegativeParams",
    "NormalizedTrace",
    "OrderingParams",
    "OutputFormatParams",
    "ParamsValidationError",
    "StateParams",
    "StatisticalParams",
    "StepType",
    "ToolCallStep",
    "ToolSelectionParams",
    "TraceIn",
    "TraceMetadata",
    "TraceStep",
    "UserInputStep",
    "schema_for",
    "validate_params",
]

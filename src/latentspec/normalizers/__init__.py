"""Stage 1 — trace normalization (§3.2).

Raw traces arrive in varying formats (LangChain runs, OpenTelemetry spans,
custom JSON). The normalizer registry converts each input to the unified
internal `NormalizedTrace` representation. Every framework integration in
§5.2 plugs in here.
"""

from latentspec.normalizers.base import Normalizer, NormalizerError, NormalizerRegistry, registry
from latentspec.normalizers.langchain import LangChainNormalizer
from latentspec.normalizers.raw_json import RawJSONNormalizer

__all__ = [
    "LangChainNormalizer",
    "Normalizer",
    "NormalizerError",
    "NormalizerRegistry",
    "RawJSONNormalizer",
    "registry",
]

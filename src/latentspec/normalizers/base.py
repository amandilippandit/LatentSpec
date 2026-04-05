"""Normalizer base class and registry.

A normalizer takes an opaque payload (whatever the agent's framework produces)
and returns a `NormalizedTrace` matching the §3.2 schema. The registry lets
the API route dispatch on a `format` discriminator without hard-coding
framework imports into request handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from latentspec.schemas.trace import NormalizedTrace


class NormalizerError(ValueError):
    """Raised when a payload cannot be normalized to the §3.2 schema."""


class Normalizer(ABC):
    """Implement this for each input format in §5.2."""

    name: str = ""

    @abstractmethod
    def normalize(self, payload: dict[str, Any], *, agent_id: str) -> NormalizedTrace:
        ...


class NormalizerRegistry:
    def __init__(self) -> None:
        self._normalizers: dict[str, Normalizer] = {}

    def register(self, normalizer: Normalizer) -> None:
        if not normalizer.name:
            raise ValueError(f"normalizer {normalizer!r} is missing a name")
        self._normalizers[normalizer.name] = normalizer

    def get(self, name: str) -> Normalizer:
        try:
            return self._normalizers[name]
        except KeyError as e:
            raise NormalizerError(f"unknown trace format: {name!r}") from e

    def normalize(
        self, format: str, payload: dict[str, Any], *, agent_id: str
    ) -> NormalizedTrace:
        return self.get(format).normalize(payload, agent_id=agent_id)


registry = NormalizerRegistry()

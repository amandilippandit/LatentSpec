"""PII redaction at the SDK boundary.

Three layers, each independently configurable:

  1. **Pattern-based regex** — fast deterministic matching for emails,
     credit cards, SSNs, API tokens, phones, IPv4 (default).
  2. **Field-key blocklist** — drop entire fields whose key matches
     `password`, `auth_token`, `api_key`, etc.
  3. **Custom redactor pipeline** — register zero-or-more user-supplied
     callables that run after pattern + blocklist redaction. Each gets
     a string and returns a string. This is the hook where customers plug
     in NER models (spaCy, Presidio, transformers), commercial DLP, or
     domain-specific patterns (medical record numbers, internal IDs).
  4. **Optional NER backend** — the `NerBackend` ABC accepts pluggable
     entity extractors. We ship one in-process backend with the lazy
     `transformers` import (no hard dep), and one HTTP backend hooking a
     remote Presidio Analyzer. Both feed `redact_string` automatically.

The whole pipeline runs left-to-right, idempotent, and short-circuits when
the trace doesn't carry strings.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Pattern

log = logging.getLogger(__name__)


# ---------------- Pattern catalog -----------------------------------------


_DEFAULT_PATTERNS: dict[str, Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "credit_card": re.compile(r"(?:\d[ -]?){13,19}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "api_token": re.compile(
        r"\b(?:sk|pk|ghp|ls)_[A-Za-z0-9_]{16,}\b|\bAKIA[0-9A-Z]{16}\b"
    ),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+?\.[A-Za-z0-9_-]+?\.[A-Za-z0-9_-]+\b"),
    "phone": re.compile(r"\+?\d{1,3}[ -.]?\(?\d{2,4}\)?[ -.]?\d{3,4}[ -.]?\d{3,4}"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ipv6": re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
}


# ---------------- NER backend interface -----------------------------------


@dataclass
class NerEntity:
    label: str
    start: int
    end: int


class NerBackend(ABC):
    """Pluggable named-entity recognition backend."""

    name: str = "ner"

    @abstractmethod
    def detect(self, text: str) -> list[NerEntity]: ...


class TransformersNerBackend(NerBackend):
    """In-process NER via the optional `transformers` package.

    Loads the model lazily on the first call. Falls back to a no-op when
    transformers isn't installed so the SDK keeps working with regex-only
    redaction. The default model is a small dslim/bert-base-NER variant —
    swap via `model_name` to bring your own.
    """

    name = "transformers"

    def __init__(
        self,
        *,
        model_name: str = "dslim/bert-base-NER",
        device: int = -1,
        labels: tuple[str, ...] = (
            "PER",
            "ORG",
            "LOC",
            "GPE",
            "MISC",
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
        ),
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._labels = {label.upper() for label in labels}
        self._pipeline: Any | None = None
        self._loaded = False

    def _load(self) -> Any | None:
        if self._loaded:
            return self._pipeline
        self._loaded = True
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "ner",
                model=self._model_name,
                aggregation_strategy="simple",
                device=self._device,
            )
        except Exception as e:  # noqa: BLE001
            log.info("transformers NER unavailable (%s); falling back to regex", e)
            self._pipeline = None
        return self._pipeline

    def detect(self, text: str) -> list[NerEntity]:
        pipe = self._load()
        if pipe is None:
            return []
        try:
            ents = pipe(text)
        except Exception as e:  # noqa: BLE001
            log.debug("NER pipeline error: %s", e)
            return []
        out: list[NerEntity] = []
        for ent in ents:
            label = str(ent.get("entity_group") or ent.get("entity") or "").upper()
            if label not in self._labels:
                continue
            out.append(
                NerEntity(
                    label=label,
                    start=int(ent.get("start", 0)),
                    end=int(ent.get("end", 0)),
                )
            )
        return out


class HttpPresidioBackend(NerBackend):
    """Remote NER via Microsoft Presidio Analyzer's HTTP API.

    Configure with the deployed analyzer URL. We only call out when
    `enabled=True` and the URL is reachable; failures are silent so a
    degraded NER service can't take down the trace pipeline.
    """

    name = "presidio"

    def __init__(self, *, url: str, timeout: float = 1.5) -> None:

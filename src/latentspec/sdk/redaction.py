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
        self._url = url.rstrip("/")
        self._timeout = timeout

    def detect(self, text: str) -> list[NerEntity]:
        try:
            import httpx

            resp = httpx.post(
                f"{self._url}/analyze",
                json={"text": text, "language": "en"},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            log.debug("presidio backend error: %s", e)
            return []
        out: list[NerEntity] = []
        for ent in data:
            out.append(
                NerEntity(
                    label=str(ent.get("entity_type", "")),
                    start=int(ent.get("start", 0)),
                    end=int(ent.get("end", 0)),
                )
            )
        return out


# ---------------- Custom-redactor pipeline --------------------------------


CustomRedactor = Callable[[str], str]


@dataclass
class Redactor:
    enabled: bool = True
    patterns: dict[str, Pattern[str]] = None  # type: ignore[assignment]
    placeholder: str = "[redacted:{name}]"
    custom_pipeline: list[CustomRedactor] = field(default_factory=list)
    field_blocklist: frozenset[str] = frozenset(
        {
            "password", "auth_token", "api_key", "authorization",
            "session_token", "private_key", "client_secret",
            "credentials", "bearer_token",
        }
    )
    ner_backends: list[NerBackend] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.patterns is None:
            self.patterns = dict(_DEFAULT_PATTERNS)

    def _apply_ner(self, value: str) -> str:
        if not self.ner_backends:
            return value
        # Collect all spans from every backend, then mask longest-first to
        # avoid index-shifting bugs.
        spans: list[tuple[int, int, str]] = []
        for backend in self.ner_backends:
            try:
                for ent in backend.detect(value):
                    spans.append((ent.start, ent.end, ent.label.lower()))
            except Exception as e:  # noqa: BLE001
                log.debug("NER backend %s failed: %s", backend.name, e)
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))

        out = value
        for start, end, label in spans:
            placeholder = self.placeholder.format(name=label)
            # Re-resolve indices in case earlier spans changed the length.
            try:
                # Replace only the first occurrence of the original substring.
                slice_text = value[start:end]
                if slice_text and slice_text in out:
                    out = out.replace(slice_text, placeholder, 1)
            except IndexError:
                continue
        return out

    def redact_string(self, value: str) -> str:
        if not value or not self.enabled:
            return value
        out = value
        # 1) Pattern-based regex
        for name, pattern in self.patterns.items():
            out = pattern.sub(self.placeholder.format(name=name), out)
        # 2) NER backends
        out = self._apply_ner(out)
        # 3) Custom pipeline
        for fn in self.custom_pipeline:
            try:
                out = fn(out)
            except Exception as e:  # noqa: BLE001
                log.debug("custom redactor failed: %s", e)
        return out

    def redact_value(self, key: str | None, value: Any) -> Any:
        if not self.enabled:
            return value
        if key is not None and key.lower() in self.field_blocklist:
            return "[redacted:blocked_field]"
        if isinstance(value, str):
            return self.redact_string(value)
        if isinstance(value, dict):
            return {k: self.redact_value(str(k), v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact_value(None, v) for v in value]
        return value

    def add_pattern(self, name: str, regex: str | Pattern[str]) -> None:
        if isinstance(regex, str):
            regex = re.compile(regex)
        self.patterns[name] = regex

    def add_custom_redactor(self, fn: CustomRedactor) -> None:
        self.custom_pipeline.append(fn)

    def add_ner_backend(self, backend: NerBackend) -> None:
        self.ner_backends.append(backend)


_default = Redactor()


def get_default_redactor() -> Redactor:
    return _default


def redact_value(key: str | None, value: Any) -> Any:
    return _default.redact_value(key, value)


def redact_string(value: str) -> str:
    return _default.redact_string(value)


def configure(
    *,
    enabled: bool | None = None,
    placeholder: str | None = None,
    custom_redactors: list[CustomRedactor] | None = None,
    field_blocklist: frozenset[str] | None = None,
    ner_backends: list[NerBackend] | None = None,
) -> None:
    if enabled is not None:
        _default.enabled = enabled
    if placeholder is not None:
        _default.placeholder = placeholder
    if custom_redactors is not None:
        _default.custom_pipeline = list(custom_redactors)
    if field_blocklist is not None:
        _default.field_blocklist = field_blocklist
    if ner_backends is not None:
        _default.ner_backends = list(ner_backends)

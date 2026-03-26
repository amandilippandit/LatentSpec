"""Synthetic booking-agent trace generator (§9 week-1 milestone).

Importable from the CLI and the seed script. The agent simulates a
flight-booking workflow with deliberately implanted behavioral rules across
seven of the eight invariant types from §3.3.
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    TraceStep,
    UserInputStep,
)


PROMPT_BANK: list[tuple[str, str]] = [
    ("US", "I want to book a flight to New York next Tuesday"),
    ("US", "Can I change my booking from JFK to LAX?"),
    ("US", "What's the price of a ticket to Chicago?"),
    ("US", "Refund my ticket please, the meeting was cancelled"),
    ("US", "Book me a flight to San Francisco for tomorrow morning"),
    ("EU", "I need a flight from London to Berlin"),
    ("EU", "Refund the Frankfurt booking, I cannot travel"),
    ("EU", "What's the cheapest flight to Madrid this week?"),
    ("EU", "Book the 8am train from Paris to Brussels"),
    ("JP", "I would like to book a flight to Tokyo, please respond in Japanese"),
    ("JP", "Cancel my Osaka booking. Reply in Japanese please."),
    ("JP", "Show me flights from Narita to Sapporo"),
    ("JP", "Refund my Kyoto trip, I'm sorry I cannot make it"),
]


def _latency(mean: float, sigma: float = 60.0, max_ms: int = 1500) -> int:
    return max(1, min(max_ms, int(random.gauss(mean, sigma))))


def _build_trace(idx: int) -> NormalizedTrace:
    segment, user_text = random.choice(PROMPT_BANK)
    is_refund = "refund" in user_text.lower() or "cancel" in user_text.lower()
    is_japanese = segment == "JP"
    is_pricing = "price" in user_text.lower() or "cheapest" in user_text.lower()

    steps: list[TraceStep] = [UserInputStep(content=user_text)]

    # Implant 1 — high-support ordering: validate_input → load_session → ...
    steps.append(
        ToolCallStep(
            tool="validate_input",
            args={"text": user_text},
            latency_ms=_latency(40),
            result_status="success",
        )
    )
    steps.append(
        ToolCallStep(
            tool="load_session",
            args={"segment": segment},
            latency_ms=_latency(60),
            result_status="success",
        )
    )

    if is_japanese:
        steps.append(
            ToolCallStep(
                tool="translate_jp",
                args={"text": user_text},
                latency_ms=_latency(80),
                result_status="success",
            )
        )

    if is_refund:
        # Implant 2 — conditional: refund/cancel → escalate_human
        steps.append(
            ToolCallStep(
                tool="escalate_human",
                args={"reason": "refund_request"},
                latency_ms=_latency(120),
                result_status="success",
            )
        )
        steps.append(
            AgentResponseStep(
                content="I've escalated your refund request to a human agent."
            )
        )
        return _wrap(idx, steps, segment, is_japanese)

    # Implant 3 — ordering: search_flights always before booking
    steps.append(
        ToolCallStep(
            tool="search_flights",
            args={"query": user_text},
            latency_ms=_latency(220),
            result_status="success",
        )
    )

    if not is_pricing and random.random() < 0.85:
        # Implant 4 — ordering: check_inventory → create_order
        steps.append(
            ToolCallStep(
                tool="check_inventory",

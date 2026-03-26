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
                args={"flight_id": f"FL{random.randint(1000, 9999)}"},
                latency_ms=_latency(140),
                result_status="success",
            )
        )
        steps.append(
            ToolCallStep(
                tool="create_order",
                args={"flight_id": f"FL{random.randint(1000, 9999)}"},
                latency_ms=_latency(180),
                result_status="success",
            )
        )

        # Implant 5 — tool selection by segment
        payments_tool = "payments_v2" if segment == "EU" else "payments_v1"
        steps.append(
            ToolCallStep(
                tool=payments_tool,
                args={"amount": random.randint(100, 900)},
                latency_ms=_latency(260),
                result_status="success" if random.random() > 0.01 else "error",
            )
        )

        # Implant 6 — statistical: book_flight succeeds reliably
        book_success = random.random() > 0.005
        steps.append(
            ToolCallStep(
                tool="book_flight",
                args={"flight_id": f"FL{random.randint(1000, 9999)}"},
                latency_ms=_latency(310),
                result_status="success" if book_success else "error",
            )
        )

        # Implant 7 — composition: notify_user always follows successful booking
        if book_success:
            steps.append(
                ToolCallStep(
                    tool="notify_user",
                    args={"channel": "email"},
                    latency_ms=_latency(90),
                    result_status="success",
                )
            )

        # Implant 8 — state: session_close terminates the trace
        steps.append(
            ToolCallStep(
                tool="session_close",
                args={},
                latency_ms=_latency(20),
                result_status="success",
            )
        )

        if is_japanese:
            steps.append(
                AgentResponseStep(
                    content="ご予約が完了しました。確認メールをお送りしました。"
                )
            )
        else:
            steps.append(
                AgentResponseStep(content="Your booking is confirmed. Check your email.")
            )
    else:
        # Implant 9 — output format: pricing answers cite live pricing data
        steps.append(
            ToolCallStep(
                tool="lookup_pricing",
                args={"route": user_text},
                latency_ms=_latency(110),
                result_status="success",
            )
        )
        steps.append(
            AgentResponseStep(
                content=(
                    "According to current pricing data, fares range from $189 to $645. "
                    "Note: prices reflect the live fare database."
                )
            )
        )

    return _wrap(idx, steps, segment, is_japanese)


def _wrap(idx: int, steps: list[TraceStep], segment: str, is_japanese: bool) -> NormalizedTrace:
    started_at = datetime.now(UTC) - timedelta(minutes=random.randint(0, 60 * 24 * 14))
    duration_ms = sum(getattr(s, "latency_ms", 0) or 0 for s in steps)
    ended_at = started_at + timedelta(milliseconds=duration_ms)

    return NormalizedTrace(
        trace_id=f"trace-{idx:05d}-{uuid.uuid4().hex[:6]}",
        agent_id="booking-agent-v2",
        timestamp=started_at,
        ended_at=ended_at,
        steps=steps,
        metadata=TraceMetadata(
            model="claude-sonnet-4-5",
            version="v2.1",
            user_segment=segment,
            locale="ja-JP" if is_japanese else "en-US",
        ),
    )


def generate_traces(n: int, *, seed: int = 42) -> list[NormalizedTrace]:
    """Produce `n` synthetic booking-agent traces with reproducible seeding."""
    random.seed(seed)
    return [_build_trace(i) for i in range(n)]

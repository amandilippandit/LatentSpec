"""Initial schema — five core tables (§8.1) plus TimescaleDB hypertable on traces.

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-01

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


INVARIANT_EMBEDDING_DIM = 1536


def upgrade() -> None:
    # Extensions are created by the init-db.sql bootstrap, but we ensure
    # idempotently here in case of fresh prod deploys via Alembic only.
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ---- Enums ---------------------------------------------------------
    trace_status = postgresql.ENUM(
        "success", "error", "timeout", "partial", name="trace_status"
    )
    invariant_type = postgresql.ENUM(
        "ordering",
        "conditional",
        "negative",
        "statistical",
        "output_format",
        "tool_selection",
        "state",
        "composition",
        name="invariant_type",
    )
    severity = postgresql.ENUM("low", "medium", "high", "critical", name="severity")
    invariant_status = postgresql.ENUM("pending", "active", "rejected", name="invariant_status")
    mining_run_status = postgresql.ENUM(
        "pending", "running", "completed", "failed", name="mining_run_status"
    )
    for enum in (trace_status, invariant_type, severity, invariant_status, mining_run_status):
        enum.create(op.get_bind(), checkfirst=True)

    # ---- agents --------------------------------------------------------
    op.create_table(
        "agents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.String(1024), nullable=True),
        sa.Column("framework", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ---- traces (TimescaleDB hypertable) ------------------------------
    op.create_table(
        "traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trace_data", postgresql.JSONB, nullable=False),
        sa.Column("version_tag", sa.String(64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("step_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "status",
            postgresql.ENUM(name="trace_status", create_type=False),
            nullable=False,
            server_default="success",
        ),
    )
    op.create_index(
        "ix_traces_agent_started", "traces", ["agent_id", "started_at"]
    )
    op.create_index("ix_traces_version", "traces", ["agent_id", "version_tag"])

    # Promote traces to a hypertable (§7 / §8.2 hot tier)
    op.execute(
        "SELECT create_hypertable('traces', 'started_at', "
        "if_not_exists => TRUE, migrate_data => TRUE)"
    )

    # ---- invariants ---------------------------------------------------
    op.create_table(
        "invariants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "type",
            postgresql.ENUM(name="invariant_type", create_type=False),
            nullable=False,
        ),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("formal_rule", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("support_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("consistency_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("cross_val_bonus", sa.Float, nullable=False, server_default="0"),
        sa.Column("clarity_score", sa.Float, nullable=False, server_default="0"),
        sa.Column(
            "severity",
            postgresql.ENUM(name="severity", create_type=False),

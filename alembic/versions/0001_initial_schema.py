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
            nullable=False,
            server_default="medium",
        ),
        sa.Column(
            "status",
            postgresql.ENUM(name="invariant_status", create_type=False),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "evidence_trace_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("evidence_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("violation_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("violation_rate", sa.Float, nullable=False, server_default="0"),
        sa.Column(
            "discovered_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("description_embedding", Vector(INVARIANT_EMBEDDING_DIM), nullable=True),
        sa.Column("discovered_by", sa.String(32), nullable=False, server_default="both"),
        sa.Column("params", postgresql.JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_invariants_agent_status", "invariants", ["agent_id", "status"])
    op.create_index("ix_invariants_agent_type", "invariants", ["agent_id", "type"])

    # ---- violations ---------------------------------------------------
    op.create_table(
        "violations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "invariant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("invariants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "trace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("traces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("details", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "severity",
            postgresql.ENUM(name="severity", create_type=False),
            nullable=False,
        ),
        sa.Column("acknowledged", sa.Boolean, nullable=False, server_default="false"),
    )
    op.create_index(
        "ix_violations_invariant_detected", "violations", ["invariant_id", "detected_at"]
    )

    # ---- mining_runs --------------------------------------------------
    op.create_table(
        "mining_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("traces_analyzed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("invariants_discovered", sa.Integer, nullable=False, server_default="0"),
        sa.Column("config", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "status",
            postgresql.ENUM(name="mining_run_status", create_type=False),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error", sa.Text, nullable=True),
    )

    # ---- §8.2 retention policies (TimescaleDB compression + drop) -----
    # 30-day hot, 90-day warm; cold tier (>90d) is paywalled enterprise feature
    # and is implemented out-of-band against object storage, not Postgres.
    op.execute(
        "ALTER TABLE traces SET ("
        "timescaledb.compress, "
        "timescaledb.compress_segmentby = 'agent_id'"
        ")"
    )
    op.execute("SELECT add_compression_policy('traces', INTERVAL '30 days')")
    op.execute("SELECT add_retention_policy('traces', INTERVAL '90 days')")


def downgrade() -> None:
    op.execute("SELECT remove_retention_policy('traces', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('traces', if_exists => TRUE)")
    op.drop_table("mining_runs")
    op.drop_index("ix_violations_invariant_detected", table_name="violations")
    op.drop_table("violations")
    op.drop_index("ix_invariants_agent_type", table_name="invariants")
    op.drop_index("ix_invariants_agent_status", table_name="invariants")
    op.drop_table("invariants")
    op.drop_index("ix_traces_version", table_name="traces")
    op.drop_index("ix_traces_agent_started", table_name="traces")
    op.drop_table("traces")
    op.drop_table("agents")
    for enum_name in (
        "mining_run_status",
        "invariant_status",
        "severity",
        "invariant_type",
        "trace_status",
    ):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")

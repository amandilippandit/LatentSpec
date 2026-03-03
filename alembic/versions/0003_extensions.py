"""Extension tables: persisted state for the seven coverage extensions.

Adds: tool_aliases, agent_versions, fingerprint_baselines,
cluster_centroids, drift_state, sessions, synthetic_review_queue,
mining_jobs, calibration_results. Also extends `traces` with
`session_id`, `user_id`, `cluster_id`, `fingerprint` columns.

Revision ID: 0003_extensions
Revises: 0002_auth
Create Date: 2026-05-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0003_extensions"
down_revision = "0002_auth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- enums -----------------------------------------------------------
    job_kind = postgresql.ENUM(
        "mining", "cluster_mining", "session_mining", "calibration",
        "synthetic_generation", "pack_autofit",
        name="job_kind",
    )
    job_status = postgresql.ENUM(
        "pending", "running", "succeeded", "failed", "cancelled",
        name="job_status",
    )
    review_decision = postgresql.ENUM(
        "pending", "approved", "rejected", "edited",
        name="review_decision",
    )
    for enum in (job_kind, job_status, review_decision):
        enum.create(op.get_bind(), checkfirst=True)

    # ---- tool_aliases ---------------------------------------------------
    op.create_table(
        "tool_aliases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("raw_name", sa.String(256), nullable=False),
        sa.Column("canonical_name", sa.String(256), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("method", sa.String(32), nullable=False, server_default="exact"),
        sa.Column("name_embedding", Vector(384), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("agent_id", "raw_name", name="uq_tool_aliases_agent_raw"),
    )
    op.create_index("ix_tool_aliases_agent_id", "tool_aliases", ["agent_id"])
    op.create_index(
        "ix_tool_aliases_canonical", "tool_aliases", ["agent_id", "canonical_name"]
    )

    # ---- agent_versions -------------------------------------------------
    op.create_table(
        "agent_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_tag", sa.String(64), nullable=False),
        sa.Column(
            "tool_repertoire",
            postgresql.ARRAY(sa.String),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("parent_version_tag", sa.String(64), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.UniqueConstraint("agent_id", "version_tag", name="uq_agent_versions_agent_tag"),
    )
    op.create_index("ix_agent_versions_agent_id", "agent_versions", ["agent_id"])

    # ---- fingerprint_baselines ------------------------------------------
    op.create_table(
        "fingerprint_baselines",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("baseline_counts", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("observed_counts", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("n_observed", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "chi_square_threshold", sa.Float, nullable=False, server_default="13.82"
        ),
        sa.Column("last_drift_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_drift_chi", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ---- cluster_centroids ----------------------------------------------
    op.create_table(
        "cluster_centroids",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cluster_id", sa.Integer, nullable=False),
        sa.Column("centroid", postgresql.JSONB, nullable=False),
        sa.Column("n_traces", sa.Integer, nullable=False, server_default="0"),
        sa.Column("silhouette", sa.Float, nullable=False, server_default="0"),
        sa.Column(
            "vectorizer_state", postgresql.JSONB, nullable=False, server_default="{}"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "agent_id", "cluster_id", name="uq_cluster_centroids_agent_cluster"
        ),
    )
    op.create_index("ix_cluster_centroids_agent_id", "cluster_centroids", ["agent_id"])

    # ---- drift_state ----------------------------------------------------
    op.create_table(
        "drift_state",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "invariant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("invariants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ph_state", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("cusum_state", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "last_updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),

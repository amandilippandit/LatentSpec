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
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "agent_id", "invariant_id", name="uq_drift_state_agent_inv"
        ),
    )
    op.create_index("ix_drift_state_agent_id", "drift_state", ["agent_id"])
    op.create_index("ix_drift_state_invariant_id", "drift_state", ["invariant_id"])

    # ---- sessions -------------------------------------------------------
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_id", sa.String(128), nullable=True),
        sa.Column("user_id", sa.String(128), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("n_turns", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_index("ix_sessions_agent_id", "sessions", ["agent_id"])
    op.create_index("ix_sessions_external_id", "sessions", ["external_id"])
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    # ---- synthetic_review_queue -----------------------------------------
    op.create_table(
        "synthetic_review_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("spec_name", sa.String(128), nullable=False),
        sa.Column("trace_data", postgresql.JSONB, nullable=False),
        sa.Column(
            "decision",
            postgresql.ENUM(name="review_decision", create_type=False),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.String(128), nullable=True),
        sa.Column("edit_notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_synthetic_review_queue_agent_id", "synthetic_review_queue", ["agent_id"]
    )
    op.create_index(
        "ix_synthetic_review_queue_decision",
        "synthetic_review_queue",
        ["decision"],
    )

    # ---- mining_jobs ----------------------------------------------------
    op.create_table(
        "mining_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "kind", postgresql.ENUM(name="job_kind", create_type=False), nullable=False
        ),
        sa.Column(
            "status",
            postgresql.ENUM(name="job_status", create_type=False),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("progress_percent", sa.Float, nullable=False, server_default="0"),
        sa.Column("progress_message", sa.String(512), nullable=True),
        sa.Column("config", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("result", postgresql.JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )
    op.create_index("ix_mining_jobs_agent_id", "mining_jobs", ["agent_id"])
    op.create_index("ix_mining_jobs_status", "mining_jobs", ["status"])

    # ---- calibration_results --------------------------------------------
    op.create_table(
        "calibration_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("n_traces_calibrated", sa.Integer, nullable=False, server_default="0"),
        sa.Column("mining_min_support", sa.Float, nullable=False, server_default="0.6"),
        sa.Column(
            "mining_min_directionality", sa.Float, nullable=False, server_default="0.9"
        ),
        sa.Column("mining_max_path_length", sa.Integer, nullable=False, server_default="3"),
        sa.Column(
            "association_min_mi_bits", sa.Float, nullable=False, server_default="0.05"
        ),
        sa.Column("association_min_lift", sa.Float, nullable=False, server_default="0.2"),
        sa.Column(
            "association_min_keyword_traces", sa.Integer, nullable=False, server_default="10"
        ),
        sa.Column("statistical_p_target", sa.Float, nullable=False, server_default="99"),
        sa.Column(
            "anomaly_contamination", sa.Float, nullable=False, server_default="0.05"
        ),
        sa.Column(
            "confidence_reject_threshold", sa.Float, nullable=False, server_default="0.6"
        ),
        sa.Column(
            "confidence_review_threshold", sa.Float, nullable=False, server_default="0.8"
        ),
        sa.Column(
            "fingerprint_chi_square_threshold",
            sa.Float,
            nullable=False,
            server_default="13.82",
        ),
        sa.Column("drift_ph_threshold", sa.Float, nullable=False, server_default="8.0"),
        sa.Column("drift_cusum_threshold", sa.Float, nullable=False, server_default="4.0"),
        sa.Column(
            "distribution_summary", postgresql.JSONB, nullable=False, server_default="{}"
        ),
        sa.Column(
            "calibrated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ---- extend traces table --------------------------------------------
    op.add_column("traces", sa.Column("session_id", sa.String(128), nullable=True))
    op.add_column("traces", sa.Column("user_id", sa.String(128), nullable=True))
    op.add_column("traces", sa.Column("cluster_id", sa.Integer, nullable=True))
    op.add_column("traces", sa.Column("fingerprint", sa.String(64), nullable=True))
    op.create_index("ix_traces_session_id", "traces", ["session_id"])
    op.create_index("ix_traces_user_id", "traces", ["user_id"])
    op.create_index("ix_traces_cluster_id", "traces", ["cluster_id"])
    op.create_index("ix_traces_fingerprint", "traces", ["fingerprint"])


def downgrade() -> None:
    op.drop_index("ix_traces_fingerprint", table_name="traces")
    op.drop_index("ix_traces_cluster_id", table_name="traces")
    op.drop_index("ix_traces_user_id", table_name="traces")
    op.drop_index("ix_traces_session_id", table_name="traces")
    op.drop_column("traces", "fingerprint")
    op.drop_column("traces", "cluster_id")
    op.drop_column("traces", "user_id")
    op.drop_column("traces", "session_id")

    op.drop_table("calibration_results")
    op.drop_table("mining_jobs")
    op.drop_table("synthetic_review_queue")
    op.drop_table("sessions")
    op.drop_table("drift_state")
    op.drop_table("cluster_centroids")
    op.drop_table("fingerprint_baselines")
    op.drop_table("agent_versions")
    op.drop_table("tool_aliases")

    for enum_name in ("review_decision", "job_status", "job_kind"):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")

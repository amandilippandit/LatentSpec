"""Centralized settings loaded from environment variables.

The pricing/policy bands here come straight from the technical plan:
- §3.4 confidence thresholds (0.6 reject / 0.8 auto-activate)
- §3.2 LLM batch size (50–100 traces)
- §8.2 hot/warm/cold retention (30 / 90 / >90 days)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://latentspec:latentspec@localhost:5432/latentspec"
    database_sync_url: str = "postgresql://latentspec:latentspec@localhost:5432/latentspec"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # Anthropic Claude API (LLM mining track + LLM-as-judge checker)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # Mining configuration (§3.2)
    mining_batch_size: int = Field(default=75, ge=10, le=200)
    mining_min_support: float = Field(default=0.6, ge=0.0, le=1.0)

    # Confidence thresholds (§3.4 three-band gating)
    confidence_reject_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence_review_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Storage tiers (§8.2)
    hot_retention_days: int = 30
    warm_retention_days: int = 90


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

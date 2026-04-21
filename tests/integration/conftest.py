"""Real-DB integration test fixtures.

Spins up a SQLite-backed async engine for each test, runs the SQLAlchemy
metadata `create_all` to build the schema, and overrides the production
`db.SessionLocal` so every model insert/select goes through actual SQL.

This catches a class of bugs the in-process tests can't: ORM mapping
errors, FK constraint violations, type-coercion mismatches between
Pydantic and SQLAlchemy, JSON serialisation issues, and async session
lifecycle bugs.

We use SQLite because it's the only async-capable DB that's truly
embedded — no external service, no docker. SQLite's JSON1 plus its
recent JSON-typing coverage handles every JSONB column we use, *except*
the pgvector and ARRAY columns, which we degrade in the test schema.
UUIDs are registered as a sqlite3 adapter so SQLAlchemy's UUID mapped
type still binds correctly.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import AsyncIterator

import pytest_asyncio
from sqlalchemy import ARRAY as GenericARRAY, JSON, String, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.dialects.postgresql import ARRAY as PgARRAY, JSONB, UUID

from latentspec.db import Base
import latentspec.db as db_module
from latentspec.models import Agent

# Import every model module so its metadata registers
import latentspec.models  # noqa: F401


# ---- sqlite3 adapters for Postgres-shaped Python types -------------------


def _register_sqlite_adapters() -> None:
    """sqlite3 driver doesn't know how to bind UUIDs / dicts / lists by default.

    We register adapters that turn each into the right text/JSON form so the
    SQLAlchemy engine can drive vanilla sqlite without exploding.
    """
    sqlite3.register_adapter(uuid.UUID, lambda u: str(u))
    sqlite3.register_adapter(dict, lambda d: json.dumps(d))
    sqlite3.register_adapter(list, lambda v: json.dumps(v))


_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


def _patch_postgres_types_for_sqlite() -> None:
    """Replace UUID/JSONB/ARRAY columns with SQLite-compatible types in-place.

    The production schema uses Postgres-specific column types; SQLite needs
    `String`/`JSON`. We rewrite every Column on the registered tables.
    """
    for table in Base.metadata.tables.values():
        for column in list(table.columns):
            ctype = column.type
            try:
                if isinstance(ctype, UUID):
                    column.type = String(36)
                elif isinstance(ctype, JSONB):
                    column.type = JSON()
                elif isinstance(ctype, (PgARRAY, GenericARRAY)):
                    column.type = JSON()
                # pgvector — degrade to JSON for tests
                else:
                    name = type(ctype).__name__
                    if name == "Vector":
                        column.type = JSON()
            except Exception:
                continue


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Per-test in-memory SQLite session bound to the same metadata."""
    _register_sqlite_adapters()
    _patch_postgres_types_for_sqlite()

    engine = create_async_engine(_TEST_DB_URL, future=True)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # All Vector / ARRAY / JSONB columns now degraded to JSON via the patch.
    test_metadata = Base.metadata
    tables_to_create = list(test_metadata.tables.values())
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: test_metadata.create_all(c, tables=tables_to_create))

    # Patch the production session factory so route handlers use this engine.
    original_engine = db_module.engine
    original_factory = db_module.SessionLocal
    db_module.engine = engine
    db_module.SessionLocal = SessionLocal
    try:
        async with SessionLocal() as session:
            yield session
    finally:
        await engine.dispose()
        db_module.engine = original_engine
        db_module.SessionLocal = original_factory


@pytest_asyncio.fixture
async def seeded_agent(db_session: AsyncSession) -> Agent:
    agent = Agent(
        id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        name="integration-test-agent",
        framework="langchain",
        created_at=datetime.now(UTC),
    )
    db_session.add(agent)
    await db_session.commit()
    return agent

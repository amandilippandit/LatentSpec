"""API-key authentication middleware (§9 week 3 day 5).

Every request that touches a tenant resource must carry an API key:
  - `Authorization: Bearer ls_…`, OR
  - `X-LatentSpec-Api-Key: ls_…`

The middleware:
  1. extracts the key,
  2. looks it up by SHA-256 hash (constant-time compare via hmac in
     `auth.api_key.verify_api_key`),
  3. attaches `request.state.org_id` and `request.state.api_key_id`,
  4. updates `last_used_at` (best-effort; failures don't block the request).

Open routes — `/health`, `/docs`, `/openapi.json`, `/redoc` — are
explicitly allow-listed. Auth is enforced when `LATENTSPEC_REQUIRE_API_KEY`
is set; for local development we run permissive by default so the demo
flow still works without an org.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select

from latentspec.auth.api_key import hash_api_key
from latentspec.db import SessionLocal
from latentspec.models import ApiKey

log = logging.getLogger(__name__)


_OPEN_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/docs/oauth2-redirect",
}


def _extract_key(request: Request) -> str | None:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    custom = request.headers.get("X-LatentSpec-Api-Key")
    if custom:
        return custom.strip() or None
    return None


def _is_open_path(path: str) -> bool:
    if path in _OPEN_PATHS:
        return True
    return path.startswith("/docs/") or path.startswith("/static/")


async def api_key_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.method in {"OPTIONS", "HEAD"} or _is_open_path(request.url.path):
        return await call_next(request)

    require = os.environ.get("LATENTSPEC_REQUIRE_API_KEY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    key = _extract_key(request)
    if not key:
        if require:
            return JSONResponse(
                {"detail": "missing API key (Bearer or X-LatentSpec-Api-Key)"},
                status_code=401,
            )
        return await call_next(request)

    if not key.startswith("ls_"):
        return JSONResponse({"detail": "malformed API key"}, status_code=401)

    digest = hash_api_key(key)
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(ApiKey).where(ApiKey.hash_hex == digest).limit(1)
            )
        ).scalar_one_or_none()

        if row is None or row.revoked_at is not None:
            return JSONResponse({"detail": "invalid or revoked API key"}, status_code=401)

        request.state.org_id = row.org_id
        request.state.api_key_id = row.id

        # Best-effort last_used_at touch (don't fail the request on DB hiccups)
        try:
            row.last_used_at = datetime.now(UTC)
            await session.commit()
        except Exception as e:  # noqa: BLE001
            await session.rollback()
            log.debug("could not stamp last_used_at: %s", e)

    return await call_next(request)

"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from latentspec import __version__
from latentspec.api.middleware.auth import api_key_middleware
from latentspec.api.routes import (
    active_learning,
    agents,
    auth,
    check,
    fingerprints,
    invariants,
    jobs,
    mining,
    observability,
    packs,
    sessions,
    traces,
    verification,
    versions,
)
from latentspec.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="LatentSpec",
        version=__version__,
        description="Discover behavioral invariants from AI agent traces.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BaseHTTPMiddleware, dispatch=api_key_middleware)

    app.include_router(traces.router, prefix="/traces", tags=["traces"])
    app.include_router(agents.router, prefix="/agents", tags=["agents"])
    app.include_router(mining.router, prefix="/agents", tags=["mining"])
    app.include_router(check.router, prefix="/agents", tags=["check"])
    app.include_router(invariants.router, prefix="/invariants", tags=["invariants"])
    app.include_router(verification.router, prefix="/invariants", tags=["verification"])
    app.include_router(observability.router, tags=["observability"])
    app.include_router(auth.router, tags=["auth"])
    app.include_router(jobs.router, tags=["jobs"])
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(packs.router, tags=["packs"])
    app.include_router(active_learning.router, tags=["active-learning"])
    app.include_router(fingerprints.router, tags=["fingerprints"])
    app.include_router(versions.router, tags=["versions"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()


def run() -> None:
    """Console entrypoint: `latentspec-api`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "latentspec.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()

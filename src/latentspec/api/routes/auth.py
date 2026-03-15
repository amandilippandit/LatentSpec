"""Org + API key management routes (§9 week-3 day-5).

  POST   /orgs                       — create organization
  GET    /orgs                       — list organizations
  POST   /orgs/{id}/api-keys         — issue a new key (returns plaintext ONCE)
  GET    /orgs/{id}/api-keys         — list keys (prefixes only)
  DELETE /api-keys/{id}              — revoke
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.auth.api_key import generate_api_key
from latentspec.db import get_db
from latentspec.models import ApiKey, Organization, PricingTier

router = APIRouter()


_SLUG_RE = re.compile(r"^[a-z0-9-]+$")


class OrgIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=2, max_length=64)
    pricing_tier: PricingTier = PricingTier.FREE


class OrgOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    pricing_tier: PricingTier
    created_at: datetime


@router.post("/orgs", response_model=OrgOut, status_code=201)
async def create_org(payload: OrgIn, db: AsyncSession = Depends(get_db)) -> Organization:
    if not _SLUG_RE.match(payload.slug):
        raise HTTPException(status_code=400, detail="slug must match [a-z0-9-]+")
    org = Organization(
        name=payload.name, slug=payload.slug, pricing_tier=payload.pricing_tier
    )
    db.add(org)
    try:
        await db.flush()
        await db.refresh(org)
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"slug already exists: {e}") from e
    return org


@router.get("/orgs", response_model=list[OrgOut])
async def list_orgs(db: AsyncSession = Depends(get_db)) -> list[Organization]:
    result = await db.execute(select(Organization).order_by(Organization.created_at.desc()))
    return list(result.scalars().all())


# ----- API keys -----------------------------------------------------------


class ApiKeyIn(BaseModel):

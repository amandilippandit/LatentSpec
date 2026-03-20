"""Multi-tenant auth scaffolding (§9 week-3 day-5).

Three primitives:
  - Organization — billing/tenant boundary; agents.org_id keys here
  - User         — member of an Organization (delegated to Clerk/Auth0 in
                    production; locally we keep a thin row for development)
  - ApiKey       — hashed API key for SDK and CI/CD use; the key itself is
                    only shown once at creation time

Pricing tier is on Organization for Stripe wiring later.
"""

from latentspec.auth.api_key import (
    APIKEY_PREFIX,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)

__all__ = [
    "APIKEY_PREFIX",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
]

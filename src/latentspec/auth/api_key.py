"""API key generation + hashing.

`ls_<32 hex chars>` — Stripe-style prefixed token. We never store the
plaintext; only the SHA-256 hash plus the first 8 chars (for display in
the dashboard so users can identify which key is which).

Hash choice: SHA-256 is fast and adequate for high-entropy random tokens.
We're not protecting low-entropy passwords, so PBKDF2/Argon2 would be
overkill.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

APIKEY_PREFIX = "ls_"
_TOKEN_BYTES = 24  # 192-bit; 48 hex chars after prefix


@dataclass(frozen=True)
class GeneratedAPIKey:
    plaintext: str
    prefix: str  # "ls_" + first 8 hex chars (safe to display)
    hash_hex: str


def generate_api_key() -> GeneratedAPIKey:
    """Generate a new API key. Returns plaintext + display-prefix + hash."""
    token = secrets.token_hex(_TOKEN_BYTES)
    plaintext = f"{APIKEY_PREFIX}{token}"
    return GeneratedAPIKey(
        plaintext=plaintext,
        prefix=plaintext[: len(APIKEY_PREFIX) + 8],
        hash_hex=hash_api_key(plaintext),
    )


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_api_key(plaintext: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(plaintext), expected_hash)

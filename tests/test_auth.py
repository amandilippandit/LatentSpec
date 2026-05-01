"""Tests for §9 week-3 day-5 auth: API key generation + hashing."""

from __future__ import annotations

from latentspec.auth.api_key import (
    APIKEY_PREFIX,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)


def test_generate_api_key_shape() -> None:
    key = generate_api_key()
    assert key.plaintext.startswith(APIKEY_PREFIX)
    # ls_ + 48 hex chars
    assert len(key.plaintext) == len(APIKEY_PREFIX) + 48
    assert key.prefix.startswith(APIKEY_PREFIX)
    assert len(key.prefix) == len(APIKEY_PREFIX) + 8
    assert len(key.hash_hex) == 64


def test_generate_keys_are_unique() -> None:
    seen = {generate_api_key().plaintext for _ in range(20)}
    assert len(seen) == 20


def test_verify_round_trip() -> None:
    key = generate_api_key()
    assert verify_api_key(key.plaintext, key.hash_hex)


def test_verify_rejects_tampered_key() -> None:
    key = generate_api_key()
    # Tamper with the last hex char
    tampered = key.plaintext[:-1] + ("0" if key.plaintext[-1] != "0" else "1")
    assert not verify_api_key(tampered, key.hash_hex)


def test_hash_is_stable() -> None:
    h1 = hash_api_key("ls_constant")
    h2 = hash_api_key("ls_constant")
    assert h1 == h2
    assert h1 != hash_api_key("ls_different")

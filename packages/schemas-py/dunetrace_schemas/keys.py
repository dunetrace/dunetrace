"""
API-key hashing — shared so ingest and the Customer API cannot disagree.

The raw secret used to be stored as the plaintext PRIMARY KEY of `api_keys`, so
any read of that table — a pg_dump, a read replica, a support query, an incident
export, a stray SELECT * in a log — handed over live working credentials for
every tenant at once. This codebase already goes to real trouble to encrypt
stored third-party credentials; its own were in the clear.

Keys are now stored only as a SHA-256 hash, alongside a short non-secret prefix
kept purely so a human can tell two keys apart in a list.

Why plain SHA-256 rather than bcrypt/argon2: an API key is a 256-bit random
token from `secrets.token_urlsafe(32)`, not a human-chosen password. There is no
dictionary to attack and no meaningful work factor to add — brute force is
already infeasible — and this hash sits on the authentication path of every
ingest request, where a deliberately slow KDF would be a throughput ceiling.
The property needed here is one-wayness, not slowness.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

KEY_PREFIX_LENGTH = 12


def generate_api_key() -> str:
    """A fresh key. 32 random bytes, url-safe, with a recognisable prefix so it
    is greppable in logs and revocable by sight."""
    return "dt_" + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_prefix(key: str) -> str:
    """The leading characters, stored in the clear to identify a key in a UI.
    Short enough to be useless as a secret."""
    return key[:KEY_PREFIX_LENGTH]


def keys_equal(a: str, b: str) -> bool:
    """Constant-time comparison, for the paths that compare two hashes in
    Python rather than in the database."""
    return hmac.compare_digest(a, b)

"""
API-key hashing — shared so ingest and the Customer API cannot disagree.

The raw secret used to be stored as the plaintext PRIMARY KEY of `api_keys`, so
any read of that table — a pg_dump, a read replica, a support query, an incident
export, a stray SELECT * in a log — handed over live working credentials for
every tenant at once. This codebase already goes to real trouble to encrypt
stored third-party credentials; its own were in the clear.

Keys are now stored only as a SHA-256 hash, alongside a short non-secret prefix
kept purely so a human can tell two keys apart in a list.

Why plain SHA-256 rather than bcrypt/argon2 — and why static analysis flagging
this as "weak password hashing" does not apply:

  * **This is not a password.** The input is always the output of
    `generate_api_key()` below: 32 bytes from `secrets.token_urlsafe`, i.e. 256
    bits of uniform entropy. It is never user-chosen and never low-entropy.
  * **Slow KDFs exist to make guessing expensive.** Their entire value is
    raising the per-guess cost against a *small* search space — a human-chosen
    password, a PIN, anything drawn from a dictionary. Against 2^256 uniformly
    random candidates there is nothing to slow down; the attacker's problem is
    already infeasible by many orders of magnitude, and no work factor changes
    that.
  * **A slow KDF here would be a real cost.** This runs on the authentication
    path of *every* ingest request. Deliberately burning ~100ms per call would
    make key verification the throughput ceiling of the whole pipeline.

The property required of this function is one-wayness — a leaked `api_keys`
table must not yield working credentials — and SHA-256 provides exactly that
for a high-entropy input.

A keyed construction (HMAC with a server-side pepper) was considered and
rejected: against a 256-bit random token it adds no meaningful resistance, while
introducing a new required secret and a rotation footgun that would invalidate
every deployed key. If keys ever become user-chosen or lower-entropy, this
reasoning stops holding and argon2id is the correct answer — that change must
happen in `generate_api_key()` and here together.
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
    """One-way digest of a high-entropy API token — see the module docstring for
    why a slow KDF is neither needed nor appropriate here.

    Static analysers classify any `key`/`secret`-named input as a password and
    flag SHA-256 accordingly. That classification is wrong for this input: it is
    always a 256-bit random token, never a human-chosen secret.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_prefix(key: str) -> str:
    """The leading characters, stored in the clear to identify a key in a UI.
    Short enough to be useless as a secret."""
    return key[:KEY_PREFIX_LENGTH]


def keys_equal(a: str, b: str) -> bool:
    """Constant-time comparison, for the paths that compare two hashes in
    Python rather than in the database."""
    return hmac.compare_digest(a, b)

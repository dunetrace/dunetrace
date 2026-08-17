"""
Encryption for customer-provided external-integration credentials at rest
(Phase 2.1). Fernet (symmetric, authenticated) via a single operator-managed
master key — no encryption-at-rest capability existed anywhere in this
codebase before this (Dunetrace's own api_keys are stored and matched in
plain text).

This service only ever ENCRYPTS: a customer submits their Langfuse public/
secret key pair once via POST /v1/orgs/integrations/langfuse, api_svc
encrypts and stores it, and never decrypts it back — not even for display;
GET returns configuration status only, never the credential. Only
integrations_worker ever decrypts (services/integrations/integrations_svc/
crypto.py), since it's the only thing that ever needs the real value to call
Langfuse's API.
"""

from __future__ import annotations

import json

from cryptography.fernet import Fernet

from api_svc.config import settings


def encrypt_credentials(credentials: dict) -> str:
    """Serializes credentials to JSON and encrypts as one opaque Fernet
    token. Raises ValueError if DUNETRACE_MASTER_KEY isn't configured — fail
    loud rather than silently storing plaintext secrets."""
    if not settings.MASTER_KEY:
        raise ValueError(
            "DUNETRACE_MASTER_KEY is not configured — cannot encrypt integration credentials."
        )
    f = Fernet(settings.MASTER_KEY.encode())
    return f.encrypt(json.dumps(credentials).encode()).decode()


def decrypt_credentials_for_webhook_verification(token: str) -> dict:
    """Deliberate, narrow exception to this module's encrypt-only rule
    (Phase 4.1). Every other credential this service encrypts (Langfuse/
    LangSmith/Braintrust API keys) is only ever decrypted by whichever
    worker actually calls that provider's API — api_svc never needs the
    plaintext back. Linear's per-org webhook signing secret is different in
    kind: verifying an inbound webhook is fundamentally "decrypt, then
    HMAC-compare," and the webhook receiver (POST /v1/webhooks/linear/{org_id})
    necessarily lives in api_svc, alongside mark_signal_resolved and every
    other webhook-shaped handler (see routers/slack.py). Not a general
    decrypt capability — used only by routers/linear_webhook.py."""
    if not settings.MASTER_KEY:
        raise ValueError("DUNETRACE_MASTER_KEY is not configured — cannot verify Linear webhooks.")
    f = Fernet(settings.MASTER_KEY.encode())
    return json.loads(f.decrypt(token.encode()).decode())

"""
Decryption for external-integration credentials (Phase 2.1). See
services/api/api_svc/crypto.py's module docstring for the full split: api_svc
only ever encrypts (on config submission); this service is the only thing
that ever decrypts, since it's the only thing that needs the real credential
to call the provider's API.
"""

from __future__ import annotations

import json

from cryptography.fernet import Fernet

from integrations_svc.config import settings


def decrypt_credentials(token: str) -> dict:
    if not settings.MASTER_KEY:
        raise ValueError(
            "DUNETRACE_MASTER_KEY is not configured — cannot decrypt integration credentials."
        )
    f = Fernet(settings.MASTER_KEY.encode())
    return json.loads(f.decrypt(token.encode()).decode())

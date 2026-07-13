"""
Decryption for per-org alert-destination credentials (Phase 4.1: Slack
webhook URL, Linear API key/webhook secret). See services/api/api_svc/
crypto.py's module docstring for the full split: api_svc only ever encrypts
(on config submission); this service is the only thing that ever decrypts a
credential to actually call Slack/Linear's API — same encrypt-only/
decrypt-only split Phase 2.1 established for external evaluation
integrations (services/integrations/integrations_svc/crypto.py).
"""

from __future__ import annotations

import json

from cryptography.fernet import Fernet

from alerts_svc.config import settings


def decrypt_credentials(token: str) -> dict:
    if not settings.MASTER_KEY:
        raise ValueError(
            "DUNETRACE_MASTER_KEY is not configured — cannot decrypt integration credentials."
        )
    f = Fernet(settings.MASTER_KEY.encode())
    return json.loads(f.decrypt(token.encode()).decode())

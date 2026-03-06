"""
services/alerts/alerts_svc/formatters/webhook.py
Signed JSON payload for generic webhooks.
"""
from __future__ import annotations
import hashlib, hmac, json, os, time

from explainer_svc.models import Explanation


def format_webhook(explanation: Explanation) -> dict:
    return {
        "schema_version": "1.0",
        "event":          "failure_signal",
        "sent_at":        time.time(),
        "failure_type":   explanation.failure_type,
        "severity":       explanation.severity,
        "confidence":     explanation.confidence,
        "run_id":         explanation.run_id,
        "agent_id":       explanation.agent_id,
        "agent_version":  explanation.agent_version,
        "step_index":     explanation.step_index,
        "detected_at":    explanation.detected_at,
        "title":            explanation.title,
        "what":             explanation.what,
        "why_it_matters":   explanation.why_it_matters,
        "evidence_summary": explanation.evidence_summary,
        "evidence":         explanation.evidence,
        "suggested_fixes": [
            {"description": f.description, "language": f.language, "code": f.code}
            for f in explanation.suggested_fixes
        ],
    }


def sign_payload(body: bytes, secret: str) -> str:
    if not secret:
        return ""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def build_signed_request(explanation: Explanation, secret: str = "") -> tuple[bytes, dict]:
    payload = format_webhook(explanation)
    body    = json.dumps(payload, separators=(",", ":")).encode()
    headers = {
        "Content-Type":        "application/json",
        "X-Dunetrace-Event":   "failure_signal",
        "X-Dunetrace-Version": "1.0",
    }
    sig = sign_payload(body, secret)
    if sig:
        headers["X-Dunetrace-Signature"] = sig
    return body, headers

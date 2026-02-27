"""
services/explainer/app/models.py

Output types for the explain layer.
An Explanation is what gets rendered in the dashboard / sent in an alert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodeFix:
    """
    A concrete, copy-pasteable code suggestion.
    language is used for syntax highlighting in the UI.
    """
    description: str
    language:    str           # "python" | "yaml" | "text"
    code:        str


@dataclass
class Explanation:
    """
    The full human-readable explanation of a FailureSignal.

    Produced by explain(signal) and consumed by:
      - The alert layer (formats into Slack/email messages)
      - The REST API (sends to dashboard)
      - Future: LLM patch suggester (uses this as context)
    """
    # Identity
    failure_type:     str
    severity:         str
    run_id:           str
    agent_id:         str
    agent_version:    str
    confidence:       float

    # The explain layer outputs — all deterministic, no LLM
    title:            str           # one-line: "Tool loop detected"
    what:             str           # one paragraph: what happened
    why_it_matters:   str           # one paragraph: impact on the user
    evidence_summary: str           # structured evidence in plain English
    suggested_fixes:  List[CodeFix] # 1–3 concrete fixes

    # Pass-through metadata
    step_index:       int
    detected_at:      float
    evidence:         Dict[str, Any] = field(default_factory=dict)

    def confidence_pct(self) -> str:
        return f"{int(self.confidence * 100)}%"

    def as_slack_text(self) -> str:
        """Compact single-message format for Slack alerts."""
        lines = [
            f":rotating_light: *{self.title}*  |  {self.severity}  |  {self.confidence_pct()} confidence",
            f"*Agent:* `{self.agent_id}` (v`{self.agent_version}`)  •  *Run:* `{self.run_id}`",
            "",
            f"*What happened:* {self.what}",
            f"*Why it matters:* {self.why_it_matters}",
            "",
            f"*Evidence:* {self.evidence_summary}",
        ]
        if self.suggested_fixes:
            lines.append("")
            lines.append(f"*Top fix:* {self.suggested_fixes[0].description}")
            if self.suggested_fixes[0].code:
                lines.append(f"```{self.suggested_fixes[0].code}```")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_type":     self.failure_type,
            "severity":         self.severity,
            "run_id":           self.run_id,
            "agent_id":         self.agent_id,
            "agent_version":    self.agent_version,
            "confidence":       self.confidence,
            "title":            self.title,
            "what":             self.what,
            "why_it_matters":   self.why_it_matters,
            "evidence_summary": self.evidence_summary,
            "suggested_fixes": [
                {
                    "description": f.description,
                    "language":    f.language,
                    "code":        f.code,
                }
                for f in self.suggested_fixes
            ],
            "step_index":    self.step_index,
            "detected_at":   self.detected_at,
            "evidence":      self.evidence,
        }

"""
External prompt store abstraction — for customer_code fixes (system prompt
edits) that the customer wants Dunetrace to push somewhere on their behalf,
rather than copy-pasting manually.

Langfuse is the only store implemented today, but it's deliberately not
special-cased in the workflow: it's one of potentially several stores a
customer might have connected (GitHub — opening a PR with the diff — is a
natural next one; see BACKLOG.md). apply-fix asks "is *a* store connected?"
via get_connected_prompt_store(), not "is Langfuse configured?" specifically.

"Connected" is a single global on/off today (LANGFUSE_PUBLIC_KEY/
LANGFUSE_SECRET_KEY env vars), matching how every other Langfuse-facing
piece of this OSS repo treats it — there's no per-org stored-credential
system to plug into (that's dunetrace-cloud territory, out of scope here).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ExternalPromptStore(ABC):
    @abstractmethod
    async def push_fix(self, prompt_name: str, fix_content: str) -> Dict[str, Any]:
        """Append fix_content to the named prompt and publish a new version.
        Returns {"new_version", "prompt_url", "old_text", "new_text"}."""
        ...


class LangfuseExternalPromptStore(ExternalPromptStore):
    """Delegates to langfuse_client.py — the actual Langfuse API calls stay
    there (exercised directly by their own tests); this class only adapts
    them to the store-agnostic interface."""

    async def push_fix(self, prompt_name: str, fix_content: str) -> Dict[str, Any]:
        from api_svc.langfuse_client import apply_langfuse_fix

        return await apply_langfuse_fix(prompt_name, fix_content)


def get_connected_prompt_store() -> Optional[ExternalPromptStore]:
    """Returns the connected store, or None if no external store is
    configured — callers must handle None (diff-only / copy-paste), not
    assume one is always available."""
    from api_svc.config import settings

    if settings.langfuse_configured:
        return LangfuseExternalPromptStore()
    return None

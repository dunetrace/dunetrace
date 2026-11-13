"""
Content caps and secret redaction for everything the SDK ships off-process.

Two independent controls, both applied in the emit hooks (``RunContext``) and
on the ``run.started`` payload (``Dunetrace.run``):

* **Caps** — every free-text field (tool args, tool output, LLM output,
  retrieval query/content, memory values, ``input_text``, ``system_prompt``) is
  cut at ``max_field_chars`` (default ``DEFAULT_MAX_FIELD_CHARS``, the same
  8192 the OTLP ingest path already enforces). A capped field carries two
  sibling wire keys, ``<field>_truncated: true`` and
  ``<field>_original_length: N``; an uncapped field carries neither, so small
  payloads are byte-identical to before.

* **Redaction** — structured tool args are walked and any value under a
  secret-looking key is replaced with ``"[REDACTED]"`` before serialisation.
  Keys are normalised to **words** — lower-cased, with ``-``/``.``/whitespace
  and camelCase boundaries all becoming ``_`` — and a key matches when its
  normalised form *equals* a denylist entry **or ends with** ``_<entry>``. So
  the built-in list catches ``Authorization``, ``X-Api-Key`` (→ ``x_api_key``),
  ``access_token``, ``accessToken``, ``clientSecret``, ``db_password``,
  ``Proxy-Authorization`` and the like without enumerating them. A customer
  hook (``Dunetrace(redact=...)``) runs *before* the denylist, so it can strip
  domain-specific fields the built-in list cannot know about.

  The camelCase split is load-bearing, not cosmetic. Normalising case and
  ``-`` alone matched ``access_token`` but missed ``accessToken``,
  ``clientSecret``, ``refreshToken``, ``authToken`` and ``apiSecret``, all of
  which then shipped in the clear — and camelCase is the dominant convention
  in the JSON tool arguments this actually sees. (``apiKey`` survived only by
  accident, via the bare ``apikey`` entry.) ``packages/sdk-ts/src/redaction.ts``
  implements the same rule; the two SDKs must redact identically.

Everything here is stdlib-only and must never raise: these functions run
inside the customer's agent process, on the instrumentation hot path.
"""

from __future__ import annotations

import json
import re
from typing import Any, FrozenSet, Iterable, Optional, Tuple

# Mirrors OTLP_MAX_ATTR_CHARS in services/ingest/ingest_svc/config.py so a run
# reads the same whichever transport it arrived on.
DEFAULT_MAX_FIELD_CHARS = 8192

REDACTED = "[REDACTED]"

# Matched against normalised keys (lower-cased, words separated by '_').
# Hyphenated entries are listed in their natural header spelling and
# normalised at compile time.
DEFAULT_DENYLIST: FrozenSet[str] = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "cookie",
        "set-cookie",
    }
)

# How deep redact_dict descends. Deep enough for any real tool payload; shallow
# enough that a pathological self-similar structure costs bounded work. Nodes
# below the limit are passed through untouched (not dropped) — the cap must
# never hide that data was sent, it just stops inspecting it.
MAX_REDACT_DEPTH = 32


# camelCase → word boundaries, in the same two passes as the TS SDK:
#   "accessToken" → "access_Token"   (lower/digit followed by upper)
#   "APIKey"      → "API_Key"        (acronym followed by a capitalised word)
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
# Header, path and prose separators are word separators too.
_SEPARATORS = re.compile(r"[-.\s]+")


def _normalise_key(key: str) -> str:
    """A key reduced to lower-case words joined by ``_``.

    ``accessToken`` → ``access_token``, ``X-Api-Key`` → ``x_api_key``,
    ``APIKey`` → ``api_key``. Matching on words is what lets one denylist
    entry cover every spelling of the same name; see the module docstring.
    """
    normalised = _CAMEL_BOUNDARY.sub(r"\1_\2", key)
    normalised = _ACRONYM_BOUNDARY.sub(r"\1_\2", normalised)
    return _SEPARATORS.sub("_", normalised).lower()


def compile_denylist(extra_keys: Optional[Iterable[str]] = None) -> Tuple[FrozenSet[str], tuple]:
    """Build the (exact-match set, suffix tuple) pair the matcher uses.

    ``extra_keys`` extends ``DEFAULT_DENYLIST``; both sides are normalised the
    same way keys are at match time, so ``redact_keys=["X-Session-Id"]``,
    ``["xSessionId"]`` and ``["x_session_id"]`` are all the same denylist. The suffix tuple is
    ``("_" + entry)`` for every entry, which is what makes ``access_token``
    match ``token`` and ``x_api_key`` match ``api_key``.
    """
    entries = {_normalise_key(k) for k in DEFAULT_DENYLIST}
    if extra_keys:
        for k in extra_keys:
            if isinstance(k, str) and k.strip():
                entries.add(_normalise_key(k.strip()))
    exact = frozenset(entries)
    # Sorted for a stable tuple; str.endswith takes the whole tuple in one call.
    suffixes = tuple(sorted("_" + e for e in exact))
    return exact, suffixes


_DEFAULT_COMPILED = compile_denylist()


def key_is_sensitive(key: Any, compiled: Optional[Tuple[FrozenSet[str], tuple]] = None) -> bool:
    """True when ``key`` names a value that must not leave the process.

    Non-string keys never match — an int or tuple key is structure, not a
    header name, and stringifying it would only invite false positives.
    """
    if not isinstance(key, str):
        return False
    exact, suffixes = compiled if compiled is not None else _DEFAULT_COMPILED
    nk = _normalise_key(key)
    return nk in exact or nk.endswith(suffixes)


def redact_dict(
    obj: Any,
    denylist: Any = None,
    *,
    max_depth: int = MAX_REDACT_DEPTH,
) -> Any:
    """Return a copy of ``obj`` with every value under a sensitive key replaced
    by ``REDACTED``.

    Walks dicts recursively, and lists/tuples (of anything — a list of header
    dicts is the common case). ``denylist`` may be an iterable of extra key
    names (merged into the defaults) or a pre-compiled pair from
    ``compile_denylist``; ``None`` means the built-in list. Never mutates its
    input, never descends past ``max_depth``, and never raises: on any
    internal failure the *original* object is returned so the hook still
    ships something rather than blocking the agent.
    """
    try:
        if denylist is None:
            compiled = _DEFAULT_COMPILED
        elif (
            isinstance(denylist, tuple)
            and len(denylist) == 2
            and isinstance(denylist[0], frozenset)
        ):
            compiled = denylist
        else:
            compiled = compile_denylist(denylist)
        return _redact_node(obj, compiled, max_depth)
    except Exception:
        return obj


def _redact_node(node: Any, compiled: Tuple[FrozenSet[str], tuple], depth: int) -> Any:
    if depth <= 0:
        return node
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if key_is_sensitive(k, compiled):
                out[k] = REDACTED
            else:
                out[k] = _redact_node(v, compiled, depth - 1)
        return out
    if isinstance(node, list):
        return [_redact_node(v, compiled, depth - 1) for v in node]
    if isinstance(node, tuple):
        return tuple(_redact_node(v, compiled, depth - 1) for v in node)
    return node


def cap_text(text: Any, max_chars: Optional[int]) -> Tuple[str, bool, int]:
    """``(text, truncated, original_length)`` for a plain string.

    ``max_chars`` of ``None`` or ``<= 0`` disables the cap. A non-str input is
    coerced with ``str()`` first (``repr``-free: the hooks only pass strings
    here, this is belt-and-braces for direct callers).
    """
    if not isinstance(text, str):
        text = _safe_str(text)
    n = len(text)
    if max_chars is not None and max_chars > 0 and n > max_chars:
        return text[:max_chars], True, n
    return text, False, n


def serialize_capped(value: Any, max_chars: Optional[int]) -> Tuple[str, bool, int]:
    """``(text, truncated, original_length)`` for a structured value.

    Serialised with ``json.dumps(default=str, ensure_ascii=False)`` so the wire
    carries JSON — the format the TS SDK already sends and every server-side
    parser tries first — with ``str()`` standing in for anything JSON can't
    represent natively (datetimes, dataclasses, custom classes). If even that
    fails (a circular reference, a tuple key) it falls back to ``repr``, and
    to a type-name placeholder if ``repr`` itself raises.
    """
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = _safe_str(value)
    return cap_text(text, max_chars)


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        try:
            return repr(value)
        except Exception:
            return f"<{type(value).__name__}>"


def put_capped(payload: dict, field: str, value: Any, max_chars: Optional[int]) -> Any:
    """Store ``value`` under ``payload[field]``, capped when it is a string, and
    set the ``<field>_truncated`` / ``<field>_original_length`` markers only
    when a cut actually happened. Returns what was stored, which is also what
    the in-process ``RunState`` should hold so local detectors and the server
    see the same text.

    Non-string values are stored untouched — the hooks type these fields as
    ``str`` but callers pass what they have, and changing the type of an
    already-odd value is not this layer's job.
    """
    if not isinstance(value, str):
        payload[field] = value
        return value
    text, truncated, n = cap_text(value, max_chars)
    payload[field] = text
    if truncated:
        payload[field + "_truncated"] = True
        payload[field + "_original_length"] = n
    return text

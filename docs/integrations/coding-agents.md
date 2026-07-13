# Coding Agent Integrations (Claude Code, Cursor, Codex)

Dunetrace exposes agent signals, run details, and health scores directly to
coding agents via the [Model Context Protocol](https://modelcontextprotocol.io)
— so a coding agent debugging your codebase can pull real production failure
data without you copy-pasting it in.

This page is a short index; the full reference — install steps for each
client, all 10 tools with example queries, worked examples, and typical
workflows (investigating a Slack alert, a cross-agent audit, a pre-deploy
check) — lives in **[docs/mcp-server.md](../mcp-server.md)**. This page isn't
a duplicate of that content; it exists so `docs/integrations/` has a
consistent entry for every integration type (alongside
[external-evaluation.md](external-evaluation.md), [github-app.md](github-app.md),
and [auto-instrumentation.md](auto-instrumentation.md)).

## Supported clients

| Client | Setup |
|---|---|
| **Claude Code** | `pip install dunetrace-mcp` registers it automatically in `~/.claude.json`. Restart Claude Code. |
| **Cursor** | Add `.cursor/mcp.json` to your project — see [docs/mcp-server.md#cursor](../mcp-server.md#cursor). |
| **Codex / OpenAI Responses API** | Run `dunetrace-mcp --sse` and point the tool endpoint at it — see [docs/mcp-server.md#codex--sse-clients](../mcp-server.md#codex--sse-clients). |

## What it's for

Ask your editor things a stack trace can't answer: *"is my agent healthy?"*,
*"what's failed in the last 24 hours?"*, *"is this Tool Loop systemic or a
one-off?"*, *"walk me through this run step by step."* All 10 tools are
read-only against your existing Dunetrace deployment — see
[docs/mcp-server.md](../mcp-server.md) for the full tool reference and
worked examples.

# Detector packs

A **detector pack** is a first-party bundle of detectors for a specific class
of agent, activated per org as a unit. Packs let Dunetrace ship detection for
specialized domains (voice, and more over time) without every org paying the
false-positive cost of detectors that don't apply to them.

| Pack | Detectors | For |
|---|---|---|
| [`voice`](./voice.md) | 9 | real-time voice agents (STT / TTS / VAD / turn-taking) |

---

## How packs relate to the rest of detection

There are three kinds of detector in Dunetrace; packs are the middle one:

- **Built-in detectors** — the always-on battery (23 of them) that runs for
  every org on every completed run. Packs never change these; they always run.
  See [detectors.md](../detectors.md).
- **Pack detectors** — first-party, Dunetrace-owned, activated per org as a
  whole. This page.
- **Custom detectors** — *your* detector logic, either written in plain
  English (translated to a config) or dropped in as a Python class. Distinct
  from packs: those are your code, packs are ours. See
  [detectors.md](../detectors.md#custom-detectors).

Two properties hold by design:

- **Built-ins always run**, pack or no pack. Activating a pack only *adds*
  detectors; it never disables or replaces a built-in.
- **A pack is binary per org** — the whole pack is on or off. There are no
  per-detector toggles within a pack.

New pack detectors, like any new detector, start in **shadow mode**: they
evaluate and show up in the dashboard's shadow view but don't fire alerts
until an operator promotes them.

---

## Activation

The org is always derived from your API key, **never** from the URL — you
cannot activate a pack for a different org by editing a request path.

**SDK:**

```python
dt.enable_pack("voice")
dt.disable_pack("voice")
dt.enabled_packs()          # -> ["voice"]
```

`enable_pack`/`disable_pack` call the Customer API, so the client needs
`api_url` (or the `DUNETRACE_API_URL` env var) and an `api_key`.

**API:**

```
POST   /v1/orgs/packs/{pack_name}    activate for this org (org from API key)
DELETE /v1/orgs/packs/{pack_name}    deactivate for this org
GET    /v1/orgs/packs                list this org's activated packs
GET    /v1/packs                     list all available packs (no org context)
```

**Dashboard:** the **Packs** page lists every available pack with an
activate/deactivate control.

Activation takes effect on the detector worker within ~60 seconds (a per-org
cache TTL), no restart required.

---

## Available packs

- **[voice](./voice.md)** — nine detectors for real-time voice agents:
  transcription confidence, silence timeouts, turn-taking collisions,
  latency-induced hangups, audio quality degradation, speaker confusion,
  barge-in handling, TTS truncation, and VAD false triggers. Ships with four
  new SDK event hooks and three voice-specific policy actions.

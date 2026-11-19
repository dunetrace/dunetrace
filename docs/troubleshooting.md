# Troubleshooting / FAQ

Common issues when getting started with Dunetrace and how to resolve them.

## I instrumented my agent but no events show up

**Why:** The SDK is not reaching the ingest API, or the backend is not running.

**Fix:**
- Confirm the backend is up: curl -s localhost:8001/ready should return {"status":"ok"}.
- If the SDK is running on a different machine/container than the backend, set DUNETRACE_ENDPOINT to the reachable ingest URL (default http://localhost:8001).
- Check for an API key if not in local auth-off mode. See the 401 section below.
- Look at the ingest service logs for rejected or dropped events.

## My run completed but I see no signals

**Why:** Most detectors ship in shadow mode by default. Shadow signals are stored but do not generate alerts.

**Fix:**
- View shadow signals in the dashboard or via the API with ?include_shadow=true.
- To promote a detector to live, set it in LIVE_DETECTORS. See detectors.md.

## A detector isn't firing

**Why:** Missing instrumentation data, thresholds not met, or agent-category overrides.

**Fix:**
- Ensure relevant events are instrumented (e.g. tool_responded).
- Check thresholds in detectors.yml.
- Look for agent-category overrides.

## 401 / auth errors

**Why:** Authentication required but not provided, or mode mismatch.

**Fix:**
- Locally with default docker-compose, auth is off (AUTH_MODE=dev).
- In production, provide a valid API key.

## Events are delayed

**Why:** Detector and alerts workers poll on an interval.

**Fix:**
- Expect 5-10 seconds of delay under normal load.

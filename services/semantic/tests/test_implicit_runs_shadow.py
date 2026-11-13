"""Semantic findings on an implicit run are stored in shadow, never alerted.

An implicit run's boundary is a guess (it can span several requests on one
thread), so the structural detector already shadows its signals; the semantic
worker has to agree, or a guessed boundary pages someone after all.
"""

from semantic_svc.worker import _is_implicit_run


def _started(payload):
    return {"event_type": "run.started", "payload": payload}


def test_implicit_run_is_detected():
    assert _is_implicit_run([_started({"implicit": True}), {"event_type": "llm.called"}])


def test_ordinary_and_entry_runs_are_not():
    assert not _is_implicit_run([_started({})])
    assert not _is_implicit_run([_started({"opened_by": "http.request"})])
    assert not _is_implicit_run([{"event_type": "llm.called"}])


def test_json_string_payload_is_read():
    assert _is_implicit_run([_started('{"implicit": true}')])
    assert not _is_implicit_run([_started("not json")])

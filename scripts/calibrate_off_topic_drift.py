#!/usr/bin/env python3
"""
Calibration harness for the OFF_TOPIC_DRIFT evaluator (Phase 8).

Run-level. Scores a labeled synthetic dataset with the real evaluator, sweeps the
GEval threshold offline, reports FP / TP recall. Ships where FP < 15%.

The FP-risk negatives are BROAD / multi-faceted questions answered thoroughly:
the response covers many aspects, which can look like wandering but is not drift
(the aspects are all related to what was asked). These are folded into the
cycled negative pool alongside plainly on-topic answers.

Usage:
    PYTHONPATH=packages/sdk-py:services/explainer:services/semantic \
      python scripts/calibrate_off_topic_drift.py [--n-per-class 50] [--reanalyze]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# (question, DRIFTING answer, ON-TOPIC answer). Drift = starts right, wanders off.
_TOPICS = [
    (
        "How much is the Pro plan?",
        "The Pro plan is $40/user/month. Speaking of value, our AI roadmap is going to transform how teams work — wait till you see next quarter!",
        "The Pro plan is $40 per user per month billed annually, or $48 billed monthly.",
    ),
    (
        "Why is my API returning 429?",
        "A 429 is rate limiting. Anyway, have you seen our enterprise tier? SSO, dedicated support, a 99.99% SLA — companies love it.",
        "A 429 means you exceeded the 60 requests/minute limit. Space out calls or request a higher limit.",
    ),
    (
        "How do I rotate my API key?",
        "Settings > API Keys > Rotate. On a related note, security matters so much now — cyberattacks rose 40% last year, everyone should use a password manager!",
        "Go to Settings > API Keys and click Rotate next to the key. The old key stops working after 24 hours.",
    ),
    (
        "What's the max file upload size?",
        "Uploads are capped at 10MB. By the way, our mobile app just got a huge redesign with dark mode and new gestures!",
        "The maximum upload size is 10MB per file. Larger files should be split or sent via the bulk API.",
    ),
    (
        "Does the export include timestamps?",
        "Yes, exports include ISO-8601 timestamps. Honestly, data is the new oil — the winners treat analytics as a first-class discipline.",
        "Yes, every exported row includes an ISO-8601 timestamp in the created_at column.",
    ),
    (
        "How do I invite a teammate?",
        "Members > Invite, enter their email. That reminds me — collaboration is the future of work, and remote teams are reshaping company culture everywhere.",
        "Go to Members > Invite, enter their email, pick a role, and send. They'll get an email to join.",
    ),
]

# Hard negatives: BROAD questions answered with many RELATED aspects. Not drift.
_BROAD_MULTI = [
    (
        "Tell me everything about the Pro plan.",
        "Pro is $40/user/month. It includes SSO, audit logs, and priority support, with a 25-seat cap and a 10MB upload limit, and you can upgrade from Basic anytime.",
    ),
    (
        "Walk me through setting up alerting.",
        "Connect Slack in Settings > Alerts, set a severity threshold, choose which detectors route where, optionally add a PagerDuty webhook, and test with 'Send test alert'.",
    ),
    (
        "What should I know before migrating?",
        "Back up your data, check API version compatibility, schedule during low traffic, review the breaking-changes list, and run the migration script in dry-run first.",
    ),
    (
        "Give me an overview of the security model.",
        "Data is encrypted in transit and at rest, access is role-based, API keys are rotatable, audit logs cover every change, and SSO is available on Pro and above.",
    ),
]


def _build_dataset(n_per_class: int):
    drift, on_topic = [], []
    i = 0
    # Broad-multi hard negatives are part of the cycled on-topic pool so FP is
    # measured against them at real weight.
    on_topic_pool = [(q, ok) for (q, _d, ok) in _TOPICS] + _BROAD_MULTI
    while len(drift) < n_per_class or len(on_topic) < n_per_class:
        q, drifting, _ok = _TOPICS[i % len(_TOPICS)]
        if len(drift) < n_per_class:
            drift.append(("drift", q, drifting))
        if len(on_topic) < n_per_class:
            oq, oa = on_topic_pool[i % len(on_topic_pool)]
            on_topic.append(("on_topic", oq, oa))
        i += 1
    return drift + on_topic


def _score_dataset(dataset, model_name):
    from semantic_svc.evaluators.base import EvaluationInput
    from semantic_svc.evaluators.off_topic_drift import OffTopicDriftEvaluator

    evaluator = OffTopicDriftEvaluator(provider="openai", model_name=model_name)
    scored = []
    for idx, (label, inp, out) in enumerate(dataset):
        result = evaluator.evaluate(EvaluationInput(input_text=inp, actual_output=out))
        score = round(1.0 - result.confidence, 4)
        scored.append({"label": label, "score": score, "confidence": result.confidence})
        print(f"  [{idx + 1}/{len(dataset)}] {label:9s} score={score:.2f}  {result.reasoning[:64]}")
    return scored


def _sweep(scored):
    drift = [s["score"] for s in scored if s["label"] == "drift"]
    ok = [s["score"] for s in scored if s["label"] == "on_topic"]
    print(f"\n{len(drift)} drift / {len(ok)} on-topic scored\n")
    header = f"{'threshold':>9} {'FP rate':>9} {'TP recall':>10} {'ship?':>6}"
    print(header)
    print("-" * len(header))
    best = None
    for t in [round(x * 0.05, 2) for x in range(6, 15)]:
        fp = sum(1 for s in ok if s < t) / max(1, len(ok))
        tp = sum(1 for s in drift if s < t) / max(1, len(drift))
        ship = fp < 0.15
        print(f"{t:>9.2f} {fp:>8.0%} {tp:>9.0%} {'yes' if ship else '':>6}")
        if ship and (best is None or tp > best[2]):
            best = (t, fp, tp)
    print("-" * len(header))
    if best:
        print(
            f"\nRECOMMENDED threshold = {best[0]:.2f}  (FP {best[1]:.0%} < 15%, TP recall {best[2]:.0%})"
        )
    else:
        print("\nNo threshold kept FP < 15% — iterate the criteria wording before shipping.")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--reanalyze", action="store_true")
    args = ap.parse_args()

    cache = pathlib.Path("scripts/calibration/off_topic_drift_scores.json")
    if args.reanalyze and cache.exists():
        print("Re-analyzing cached scores (no LLM calls).")
        _sweep(json.loads(cache.read_text()))
        return

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        sys.exit("No LLM API key — set OPENAI_API_KEY (or use --reanalyze).")

    dataset = _build_dataset(args.n_per_class)
    print(f"Scoring {len(dataset)} samples with {args.model}…")
    scored = _score_dataset(dataset, args.model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scored, indent=2))
    print(f"\nCached scores -> {cache}")
    _sweep(scored)


if __name__ == "__main__":
    main()

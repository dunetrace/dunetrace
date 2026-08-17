#!/usr/bin/env python3
"""
Calibration harness for the TASK_UNDERSTANDING_FAILURE evaluator (Phase 6).

Run-level (single input/output pairs, unlike the conversation-level CONFUSION_LOOP
harness). Scores a labeled synthetic dataset with the real evaluator, then sweeps
the GEval threshold offline and reports false-positive rate (negatives that fire)
and true-positive recall (wrong-task cases that fire). Ships at the threshold
where FP < 15% with best recall.

The negatives deliberately include the cases that separate this evaluator from
its neighbors:
  - right task but only PARTIALLY done  (TASK_COMPLETION's job, not this one)
  - genuinely AMBIGUOUS request, reasonable interpretation picked (not a failure)
  - MULTI-PART request, all parts answered (not a failure)

Usage:
    PYTHONPATH=packages/sdk-py:services/explainer:services/semantic \
      python scripts/calibrate_task_understanding_failure.py [--n-per-class 50] [--reanalyze]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# (user request, WRONG-task response, correct RIGHT-task response)
_TOPICS = [
    (
        "How much does the Pro plan cost?",
        "The Pro plan includes SSO, audit logs, and priority support.",
        "The Pro plan is $40 per user per month, billed annually.",
    ),
    (
        "Schedule a meeting with Sam for Thursday at 2pm.",
        "Effective meetings need a clear agenda and a timekeeper to stay on track.",
        "Done — I've scheduled a meeting with Sam for Thursday at 2:00pm.",
    ),
    (
        "What's the status of my refund?",
        "Our refund policy allows returns within 30 days of purchase.",
        "Your refund of $60 was issued on July 2nd and should post within 5 business days.",
    ),
    (
        "Reset my password.",
        "For security, use a long passphrase with symbols and enable two-factor auth.",
        "I've sent a password reset link to your email; it's valid for 30 minutes.",
    ),
    (
        "Cancel my subscription.",
        "Before you go — here are some features you might have missed!",
        "I've cancelled your subscription; you keep access until the period ends.",
    ),
    (
        "What's the rate limit on the /search endpoint?",
        "Our API is RESTful and returns JSON over HTTPS.",
        "The /search endpoint allows 60 requests per minute per API key.",
    ),
    (
        "Translate 'good morning' into French.",
        "French is a Romance language spoken across many countries.",
        "'Good morning' in French is 'bonjour'.",
    ),
    (
        "Book a table for four at 7pm tonight.",
        "Here are the top-rated restaurants in your neighborhood.",
        "I've booked a table for four at 7:00pm tonight under your name.",
    ),
    (
        "Summarize this document for me.",
        "Well-structured documents use clear headings and short paragraphs.",
        "Summary: the document proposes a Q3 budget increase of 12% for infrastructure.",
    ),
    (
        "Fix the null-pointer bug in this function.",
        "Unit testing early helps you catch bugs like this before they ship.",
        "The bug is on line 7 — guard `user` before dereferencing; here's the patched function.",
    ),
]

# Hard negatives (all RIGHT-task, must NOT fire), each (input, output):
_HARD_PARTIAL = [  # right task, only partially done -> TASK_COMPLETION's concern
    (
        "Book a flight to Paris and a hotel near the Louvre.",
        "I booked the flight to Paris. I couldn't complete the hotel booking — no availability near the Louvre.",
    ),
    (
        "Export my contacts and email them to me.",
        "I've exported your contacts to CSV. I wasn't able to send the email; your outbox is offline.",
    ),
    (
        "Summarize these three reports.",
        "Here are summaries of the first two reports. The third file wouldn't open.",
    ),
]
_HARD_AMBIGUOUS = [  # ambiguous request, reasonable interpretation -> not a failure
    (
        "What's the weather?",
        "In San Francisco it's currently 62°F and sunny. Let me know if you meant a different city.",
    ),
    (
        "How do I set it up?",
        "Assuming you mean the Slack integration you just added: paste the webhook URL in Settings > Alerts.",
    ),
    (
        "Is it available?",
        "The Pro plan is available in your region. If you meant a specific feature, tell me which.",
    ),
]
_HARD_MULTIPART = [  # multi-part request, all parts answered -> not a failure
    (
        "What's the Pro price and does it include SSO?",
        "The Pro plan is $40/user/month, and yes, it includes SSO.",
    ),
    (
        "Cancel my order and tell me when I'll be refunded.",
        "I've cancelled order 8842. Your refund will post within 5 business days.",
    ),
]


def _build_dataset(n_per_class: int):
    """Return list of (label, input_text, actual_output). label in
    {'wrong_task','right_task'}. All hard negatives are 'right_task'."""
    wrong, right = [], []
    i = 0
    while len(wrong) < n_per_class or len(right) < n_per_class:
        req, wrong_resp, right_resp = _TOPICS[i % len(_TOPICS)]
        if len(wrong) < n_per_class:
            wrong.append(("wrong_task", req, wrong_resp))
        if len(right) < n_per_class:
            right.append(("right_task", req, right_resp))
        i += 1
    for inp, out in _HARD_PARTIAL + _HARD_AMBIGUOUS + _HARD_MULTIPART:
        right.append(("right_task", inp, out))
    return wrong + right


def _score_dataset(dataset, model_name):
    from semantic_svc.evaluators.base import EvaluationInput
    from semantic_svc.evaluators.task_understanding_failure import (
        TaskUnderstandingFailureEvaluator,
    )

    evaluator = TaskUnderstandingFailureEvaluator(provider="openai", model_name=model_name)
    scored = []
    for idx, (label, inp, out) in enumerate(dataset):
        result = evaluator.evaluate(EvaluationInput(input_text=inp, actual_output=out))
        score = round(1.0 - result.confidence, 4)
        scored.append({"label": label, "score": score, "confidence": result.confidence})
        print(
            f"  [{idx + 1}/{len(dataset)}] {label:11s} score={score:.2f}  {result.reasoning[:64]}"
        )
    return scored


def _sweep(scored):
    wrong = [s["score"] for s in scored if s["label"] == "wrong_task"]
    right = [s["score"] for s in scored if s["label"] == "right_task"]
    print(f"\n{len(wrong)} wrong-task / {len(right)} right-task scored\n")
    header = f"{'threshold':>9} {'FP rate':>9} {'TP recall':>10} {'ship?':>6}"
    print(header)
    print("-" * len(header))
    best = None
    for t in [round(x * 0.05, 2) for x in range(6, 15)]:
        fp = sum(1 for s in right if s < t) / max(1, len(right))
        tp = sum(1 for s in wrong if s < t) / max(1, len(wrong))
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

    cache = pathlib.Path("scripts/calibration/task_understanding_failure_scores.json")
    if args.reanalyze and cache.exists():
        print(f"Re-analyzing cached scores (no LLM calls).")
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

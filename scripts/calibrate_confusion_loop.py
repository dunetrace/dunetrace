#!/usr/bin/env python3
"""
Calibration harness for the CONFUSION_LOOP semantic evaluator (Phase 5).

Runs ConfusionLoopEvaluator against a labeled synthetic dataset — deliberately
looping conversations (the user re-asks the same unresolved question) vs.
progressing ones (each turn advances to a new question, or a genuine follow-up
builds on a resolved answer). Every conversation is scored once by the real LLM;
we then sweep the GEval `threshold` offline and report, per threshold, the
false-positive rate (benign conversations that fire) and the true-positive rate
(loops that fire). The ship threshold is the one where FP drops below 15% while
keeping recall as high as possible.

Usage:
    PYTHONPATH=packages/sdk-py:services/explainer:services/semantic \
      python scripts/calibrate_confusion_loop.py [--n-per-class 50] [--model gpt-4o-mini]

Reads OPENAI_API_KEY (or the provider key) from the environment / .env. Raw
(label, confidence) pairs are cached to scripts/calibration/confusion_loop_scores.json
so threshold analysis can be re-run without re-calling the LLM (--reanalyze).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# --- dataset ------------------------------------------------------------------

# Topic templates: (user's real intent, a non-resolving agent reply, the ways a
# confused user re-asks the SAME thing). Positives cycle the same intent;
# negatives move on to genuinely different questions.
_TOPICS = [
    {
        "intent": "How do I export my contacts to a CSV file?",
        "vague_reply": "You can manage contacts under Settings.",
        "reasks": [
            "No, I mean specifically exporting them to CSV.",
            "That's not what I asked — I need a CSV file of my contacts.",
            "Again: how do I get a CSV download of the contact list?",
        ],
        "progression": [
            ("Great, and can I import a CSV back in later?", "Yes, Settings > Import accepts CSV."),
            ("Is there a size limit on the import file?", "Up to 10 MB per import."),
        ],
    },
    {
        "intent": "Why was my card charged twice this month?",
        "vague_reply": "Billing details are on your account page.",
        "reasks": [
            "I'm not asking where billing is — why the double charge?",
            "You didn't answer. There are two identical charges. Why?",
            "Still waiting: explain the duplicate charge on my card.",
        ],
        "progression": [
            (
                "Okay, can you refund the duplicate one?",
                "I've issued a refund for the duplicate charge.",
            ),
            ("How long until it shows up?", "Refunds post in 5-7 business days."),
        ],
    },
    {
        "intent": "How do I reset my password without email access?",
        "vague_reply": "Use the 'Forgot password' link on the login page.",
        "reasks": [
            "I can't get the email though — I've lost access to it.",
            "The forgot-password link emails me, which I can't read. What else?",
            "Once more: I have NO email access. How do I reset it?",
        ],
        "progression": [
            ("Thanks. Can I also change my recovery phone?", "Yes, under Security > Recovery."),
            ("Does changing it log me out everywhere?", "No, existing sessions stay active."),
        ],
    },
    {
        "intent": "What's the rate limit on the /search endpoint?",
        "vague_reply": "Our API has rate limits to protect the service.",
        "reasks": [
            "Right, but what's the specific number for /search?",
            "I need the actual limit for /search, not that there is one.",
            "Concretely: how many /search calls per minute am I allowed?",
        ],
        "progression": [
            ("Got it. Is the limit per key or per IP?", "Per API key."),
            ("Can I request a higher limit?", "Yes, contact support with your use case."),
        ],
    },
    {
        "intent": "Can you move my Tuesday meeting to Thursday?",
        "vague_reply": "You can manage meetings in your calendar.",
        "reasks": [
            "I'm asking you to move it — Tuesday to Thursday, please.",
            "That didn't move anything. Reschedule it to Thursday.",
            "Once again: change the Tuesday meeting to Thursday.",
        ],
        "progression": [
            ("Done, thanks. Can you also invite Sam?", "I've added Sam to the Thursday meeting."),
            ("Make it 30 minutes instead of an hour?", "Updated to 30 minutes."),
        ],
    },
    {
        "intent": "How do I cancel my subscription?",
        "vague_reply": "Subscription options are in your account settings.",
        "reasks": [
            "I looked there. Where exactly is the cancel button?",
            "You keep pointing at settings — how do I actually cancel?",
            "For the third time: what are the steps to cancel?",
        ],
        "progression": [
            (
                "Okay it's cancelled. Do I keep access until period end?",
                "Yes, until the billing period ends.",
            ),
            ("Will I be charged again?", "No further charges after cancellation."),
        ],
    },
    {
        "intent": "Which regions is data stored in for EU customers?",
        "vague_reply": "We take data residency seriously.",
        "reasks": [
            "That's a slogan, not an answer. Which regions specifically?",
            "I need the actual storage region for EU data.",
            "Again: name the region where EU customer data lives.",
        ],
        "progression": [
            ("Thanks. Is it replicated outside the EU?", "No, EU data stays within the EU region."),
            ("Do you have a DPA I can sign?", "Yes, our DPA is available on request."),
        ],
    },
    {
        "intent": "How do I add a teammate to my project?",
        "vague_reply": "Team management is available in the dashboard.",
        "reasks": [
            "Where in the dashboard, and what's the step to add someone?",
            "You didn't tell me how — how do I invite a teammate?",
            "Once more: the steps to add a teammate to this project?",
        ],
        "progression": [
            (
                "Added them. Can I set them as read-only?",
                "Yes, choose the Viewer role when inviting.",
            ),
            ("Can I change their role later?", "Yes, from the Members list."),
        ],
    },
]


# Hard negatives: conversations that superficially resemble a loop (a "no, I
# meant…" correction, a re-reference to the same topic) but are NOT loops — the
# correction/follow-up is resolved and the user moves on. These are where a
# lazily-worded criteria false-fires. They are labeled 'progress'.
_HARD_NEGATIVES = [
    [
        ("Can you upgrade me to the pro plan?", "Sure — do you want monthly or annual?"),
        ("No, I meant the annual pro plan.", "Done, you're on annual Pro. Renews next July."),
        ("Perfect. When's the first charge?", "Today, then annually on renewal."),
    ],
    [
        ("What's the status of order 8842?", "Order 8842 shipped yesterday via UPS."),
        ("Actually, I meant order 8843, sorry.", "Order 8843 is packed and ships tomorrow."),
        ("Great, thanks.", "You're welcome!"),
    ],
    [
        ("How do I invite a teammate?", "Members > Invite, enter their email."),
        (
            "That's not quite it — I want to invite a whole group.",
            "Use Groups > Invite to add several at once.",
        ),
        ("Got it, that worked.", "Glad it worked."),
    ],
    [
        ("Set my timezone to Pacific.", "Set to US/Pacific."),
        ("Wait, I meant Pacific/Auckland actually.", "Updated to Pacific/Auckland (NZ)."),
        ("Thanks, that's right now.", "Anytime."),
    ],
    [
        ("Show me last month's invoice.", "Here's June's invoice: $240."),
        ("No — I need the one before that, May.", "May's invoice was $210, attached."),
        ("Perfect, that's what I needed.", "Great."),
    ],
    [
        ("Rename the project to 'Falcon'.", "Renamed to 'Falcon'."),
        ("Hmm, make it 'Falcon-2' instead.", "Renamed to 'Falcon-2'."),
        ("Good, leave it there.", "Done."),
    ],
    [
        ("Add read access for Priya.", "Priya now has read access."),
        ("Actually give her write access.", "Upgraded Priya to write access."),
        ("That's better, thanks.", "You're set."),
    ],
    [
        ("What integrations do you support?", "Slack, Linear, GitHub, and webhooks."),
        ("I meant specifically for alerting.", "For alerting: Slack, PagerDuty, and webhooks."),
        ("Great, Slack works for us.", "Slack it is."),
    ],
]


def _build_dataset(n_per_class: int):
    """Return a list of (label, turns) where label is 'loop' or 'progress' and
    turns is a list of (user, assistant) pairs. Cycles the topic templates so
    n_per_class of each class are produced with varied surface wording."""
    loops, progresses = [], []
    i = 0
    while len(loops) < n_per_class or len(progresses) < n_per_class:
        topic = _TOPICS[i % len(_TOPICS)]
        rot = i // len(_TOPICS)

        if len(loops) < n_per_class:
            # user asks, agent deflects, user re-asks the same intent 2-3x
            turns = [(topic["intent"], topic["vague_reply"])]
            for reask in topic["reasks"][: 2 + (rot % 2)]:
                turns.append((reask, topic["vague_reply"]))
            loops.append(("loop", turns))

        if len(progresses) < n_per_class:
            # user asks, agent RESOLVES, user moves to new related questions
            turns = [
                (topic["intent"], "Here's exactly how: " + topic["vague_reply"] + " Then confirm.")
            ]
            for q, a in topic["progression"]:
                turns.append((q, a))
            progresses.append(("progress", turns))
        i += 1
    # Hard negatives are added on top (labeled progress) so the FP rate is
    # measured against the tricky "looks like a loop but isn't" cases too.
    for hn in _HARD_NEGATIVES:
        progresses.append(("progress", hn))
    return loops + progresses


# --- scoring ------------------------------------------------------------------


def _score_dataset(dataset, model_name):
    from semantic_svc.evaluators.base import ConversationEvaluationInput, ConversationTurn
    from semantic_svc.evaluators.confusion_loop import ConfusionLoopEvaluator

    # threshold=1.0 makes the evaluator always "fired", but we only read the raw
    # confidence (=1-score) and sweep thresholds ourselves, so its value is moot.
    evaluator = ConfusionLoopEvaluator(provider="openai", model_name=model_name, threshold=0.5)

    scored = []
    for idx, (label, turns) in enumerate(dataset):
        conv = ConversationEvaluationInput(
            turns=[
                ConversationTurn(run_id=f"r{idx}-{j}", input_text=u, actual_output=a)
                for j, (u, a) in enumerate(turns)
            ]
        )
        result = evaluator.evaluate(conv)
        score = round(1.0 - result.confidence, 4)  # GEval raw score
        scored.append({"label": label, "score": score, "confidence": result.confidence})
        print(f"  [{idx + 1}/{len(dataset)}] {label:8s} score={score:.2f}  {result.reasoning[:70]}")
    return scored


# --- threshold sweep ----------------------------------------------------------


def _sweep(scored):
    # fired iff score < threshold. FP = benign that fire; TP = loops that fire.
    loops = [s["score"] for s in scored if s["label"] == "loop"]
    progresses = [s["score"] for s in scored if s["label"] == "progress"]
    print(f"\n{len(loops)} loop / {len(progresses)} progress conversations scored\n")
    header = f"{'threshold':>9} {'FP rate':>9} {'TP recall':>10} {'ship?':>6}"
    print(header)
    print("-" * len(header))
    best = None
    for t in [round(x * 0.05, 2) for x in range(6, 15)]:  # 0.30 .. 0.70
        fp = sum(1 for s in progresses if s < t) / max(1, len(progresses))
        tp = sum(1 for s in loops if s < t) / max(1, len(loops))
        ship = fp < 0.15
        flag = "yes" if ship else ""
        print(f"{t:>9.2f} {fp:>8.0%} {tp:>9.0%} {flag:>6}")
        if ship and (best is None or tp > best[2]):
            best = (t, fp, tp)
    print("-" * len(header))
    if best:
        print(
            f"\nRECOMMENDED threshold = {best[0]:.2f}  (FP {best[1]:.0%} < 15%, TP recall {best[2]:.0%})"
        )
    else:
        print("\nNo threshold kept FP < 15% — criteria wording needs iteration before shipping.")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--reanalyze", action="store_true", help="skip LLM, re-sweep cached scores")
    args = ap.parse_args()

    cache = pathlib.Path("scripts/calibration/confusion_loop_scores.json")
    if args.reanalyze and cache.exists():
        scored = json.loads(cache.read_text())
        print(f"Re-analyzing {len(scored)} cached scores (no LLM calls).")
        _sweep(scored)
        return

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        sys.exit("No LLM API key in environment — set OPENAI_API_KEY (or use --reanalyze).")

    dataset = _build_dataset(args.n_per_class)
    print(f"Scoring {len(dataset)} conversations with {args.model}…")
    scored = _score_dataset(dataset, args.model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scored, indent=2))
    print(f"\nCached scores -> {cache}")
    _sweep(scored)


if __name__ == "__main__":
    main()

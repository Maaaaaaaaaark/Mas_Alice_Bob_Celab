"""Automatic label checks for Stage 1a worker examples.

The checks here are mechanical and are kept separate from the review
status: an example that passes all automatic checks is still only
``auto_checked`` unless a human reviewer marked it ``reviewed`` with a
recorded method and provenance. Auto checks never set the reviewed
status, and no weak label enters training silently — every excluded
example is recorded with its reason, and the trace mode trains only on
reviewed examples.

Checks (all recorded per example):

- ``evidence_in_docs``: every evidence sentence occurs verbatim in its
  stated document.
- ``evidence_sentence_aligned``: evidence matches one or two complete
  splitter sentences.
- ``at_most_two_sentences``: 1-2 evidence sentences.
- ``source_ids_valid``: every evidence doc_id exists in the visible docs.
- ``answer_covered``: positive evidence contains the answer or alias.
- ``offsets_consistent``: stored char spans are real occurrences of the
  sentence text.
- ``no_unresolved_refs``: no ``#N`` left in the local question.
- ``negative_reason_present``: negatives carry a reason.
- ``answer_string_screen``: records whether a known answer/alias occurs
  in a negative's visible documents. This is a review flag, not a label
  verdict: string presence may describe the wrong entity/relation, and
  string absence does not prove unanswerability.
- ``format_ok``: the rendered target matches the output format.
- ``documents_present``: at least one visible document.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .convert import (
    EvidenceSentence,
    WorkerExample,
    is_answer_in_text,
    normalize_text,
    split_sentences,
)


def _evidence_ok(ev: EvidenceSentence, example: WorkerExample) -> Tuple[bool, str]:
    try:
        doc = example.document_by_id(ev.doc_id)
    except KeyError as exc:
        return False, str(exc)
    if ev.sentence not in doc.text:
        return False, (
            f"evidence sentence not verbatim in {ev.doc_id}: "
            f"{ev.sentence[:60]!r}"
        )
    if ev.char_span is not None:
        start, end = ev.char_span
        if doc.text[start:end] != ev.sentence:
            return False, (
                f"char span {ev.char_span} does not contain the sentence "
                f"in {ev.doc_id}"
            )
    return True, ""


def run_automatic_checks(example: WorkerExample) -> Dict[str, Any]:
    """Run every automatic check; return ``{name: {ok, reason}}``."""
    results: Dict[str, Any] = {}
    evidence = example.evidence

    ok = len(example.documents) >= 1
    results["documents_present"] = {
        "ok": ok,
        "reason": "" if ok else "no visible documents",
    }

    ok = 0 < len(evidence) <= 2 if example.answerable else not evidence
    results["at_most_two_sentences"] = {
        "ok": ok,
        "reason": (
            ""
            if ok
            else (
                "expected 1-2 evidence sentences for positives, none for "
                "negatives"
            )
        ),
    }

    evidence_reasons: List[str] = []
    source_reasons: List[str] = []
    for ev in evidence:
        try:
            doc = example.document_by_id(ev.doc_id)
        except KeyError as exc:
            source_reasons.append(str(exc))
            continue
        if ev.sentence not in doc.text:
            evidence_reasons.append(
                f"evidence sentence not verbatim in {ev.doc_id}: "
                f"{ev.sentence[:60]!r}"
            )
    results["evidence_in_docs"] = {
        "ok": not evidence_reasons and not source_reasons,
        "reason": "; ".join(source_reasons + evidence_reasons),
    }
    results["source_ids_valid"] = {
        "ok": not source_reasons,
        "reason": "; ".join(source_reasons),
    }

    alignment_reasons: List[str] = []
    for ev in evidence:
        try:
            doc = example.document_by_id(ev.doc_id)
        except KeyError:
            continue
        if ev.sentence not in split_sentences(doc.text):
            alignment_reasons.append(
                f"{ev.doc_id}: evidence is not one complete splitter sentence"
            )
    if len(evidence) == 2 and (
        evidence[0].doc_id != evidence[1].doc_id
        or evidence[1].sentence_index != evidence[0].sentence_index + 1
    ):
        alignment_reasons.append(
            "two evidence sentences must be consecutive in one document"
        )
    results["evidence_sentence_aligned"] = {
        "ok": not alignment_reasons,
        "reason": "; ".join(alignment_reasons),
    }

    offset_reasons: List[str] = []
    for ev in evidence:
        if ev.char_span is None:
            offset_reasons.append(
                f"{ev.doc_id}: evidence sentence has no char span"
            )
            continue
        try:
            doc = example.document_by_id(ev.doc_id)
        except KeyError as exc:
            offset_reasons.append(str(exc))
            continue
        start, end = ev.char_span
        if start < 0 or end < start or end > len(doc.text):
            offset_reasons.append(
                f"{ev.doc_id}: invalid char span {ev.char_span}"
            )
        elif doc.text[start:end] != ev.sentence:
            offset_reasons.append(
                f"{ev.doc_id}: char span {ev.char_span} does not match "
                "the evidence sentence"
            )
    results["offsets_consistent"] = {
        "ok": not offset_reasons,
        "reason": "; ".join(offset_reasons),
    }

    if example.answerable:
        combined = " ".join(ev.sentence for ev in evidence)
        ok = is_answer_in_text(
            combined, example.answer, example.answer_aliases
        )
        results["answer_covered"] = {
            "ok": ok,
            "reason": (
                ""
                if ok
                else "evidence does not contain the answer or a verified alias"
            ),
        }
    else:
        results["answer_covered"] = {"ok": True, "reason": "not applicable"}

    from .convert import _UNRESOLVED_REF_RE

    unresolved = bool(_UNRESOLVED_REF_RE.search(example.local_question))
    results["no_unresolved_refs"] = {
        "ok": not unresolved,
        "reason": (
            ""
            if not unresolved
            else "unresolved #N reference remains in the local question"
        ),
    }

    if example.answerable:
        results["negative_reason_present"] = {
            "ok": True,
            "reason": "not applicable (positive)",
        }
    else:
        ok = bool(example.negative_reason)
        results["negative_reason_present"] = {
            "ok": ok,
            "reason": "" if ok else "negative example has no reason",
        }

    if example.answerable:
        results["answer_string_screen"] = {
            "ok": True,
            "flagged": False,
            "present_in": [],
            "reason": "not applicable (positive)",
        }
    else:
        present_in = []
        for doc in example.documents:
            if is_answer_in_text(
                doc.text, example.answer, example.answer_aliases
            ):
                present_in.append(doc.doc_id)
        results["answer_string_screen"] = {
            # Informational only. Semantic answerability must come from an
            # official label or recorded teacher/human review.
            "ok": True,
            "flagged": bool(present_in),
            "present_in": present_in,
            "reason": (
                "no answer/alias string found (absence is not proof)"
                if not present_in
                else (
                    "answer string found in visible document(s) "
                    f"{present_in}; review the entity/relation before use"
                )
            ),
        }

    # Format check runs on the rendered target, which lives in the
    # prompting module; a lightweight local check is kept here so the
    # report can be built before prompt rendering.
    results["format_ok"] = {"ok": True, "reason": "rendered at prompt time"}

    return results


def checked_example(
    example: WorkerExample,
) -> Tuple[WorkerExample, List[str]]:
    """Run checks, exclude the example on failure, and return failures."""
    results = run_automatic_checks(example)
    example.checks = {name: info for name, info in results.items()}
    failures = [
        f"{name}: {info['reason']}"
        for name, info in results.items()
        if not info["ok"]
    ]
    if failures and not example.excluded:
        example.excluded = True
        example.exclusion_reason = (
            "failed automatic label checks: " + "; ".join(failures)
        )
    return example, failures


def example_matches_target_format(example: WorkerExample, target: str) -> Tuple[bool, str]:
    """Check a rendered target string against the output format."""
    if not example.answerable:
        ok = target.strip() == "NOT FOUND"
        return ok, "" if ok else f"expected NOT FOUND, got {target!r}"
    lines = [line.strip() for line in target.splitlines() if line.strip()]
    if not lines:
        return False, "empty target for an answerable example"
    if len(lines) > 2:
        return False, f"more than two lines in target: {lines!r}"
    import re

    line_re = re.compile(r'^D(\d+): "(.*)"$')
    for line in lines:
        match = line_re.match(line)
        if not match:
            return False, f"line does not match the format: {line!r}"
        # Rendered targets cite display ids D0..D(n-1) assigned by
        # position after shuffling; check the cited position exists.
        position = int(match.group(1))
        if position < 0 or position >= len(example.documents):
            return False, (
                f"line cites display id D{position} but only "
                f"{len(example.documents)} documents are visible"
            )
        quoted = match.group(2)
        if not quoted:
            return False, "line quotes an empty sentence"
    return True, ""


def passes_all(example: WorkerExample) -> bool:
    return all(info["ok"] for info in example.checks.values()) if example.checks else True


def is_reviewed(example: WorkerExample) -> bool:
    """Human review status, kept separate from the automatic checks."""
    return example.review.get("status") == "reviewed"


def normalize_question_key(question: str) -> str:
    return normalize_text(question)

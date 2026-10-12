"""Prompt building and output parsing for Stage 1a worker SFT.

Prompt structure (chat template roles):

- ``system``: the worker instruction (``prompts_worker_sft/``).
- ``user``: the local question plus the visible documents, each labeled
  with a display id ``D0``, ``D1``, ... assigned by POSITION.
- ``assistant`` (target): one line per evidence sentence,
  ``D<position>: "exact sentence"``, or exactly ``NOT FOUND``.

Document order is shuffled per prompt with a fixed seed and the display
ids are remapped to the shuffled positions; the target is rendered from
the same mapping so its source ids always point at the right document.
Stable ids (``d0`` ...) exist only inside stored examples. The answer
and aliases never enter the prompt.

Output parsing/evaluation: format, source validity, verbatim quote
matching, answer coverage, and abstention are all computed here so the
same checks run on generated outputs and on the training targets.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .convert import WorkerDocument, WorkerExample, normalize_text

NOT_FOUND = "NOT FOUND"

_TARGET_LINE_RE = re.compile(r'^D(\d+): "([^"]*)"$')


def load_instruction(prompt_dir: Any) -> str:
    path = prompt_dir / "worker_system.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def shuffle_and_remap(
    example: WorkerExample, seed: int
) -> Tuple[List[WorkerDocument], Dict[str, int]]:
    """Shuffle the documents and map stable ids to display positions.

    Returns ``(ordered_docs, position_by_doc_id)`` where
    ``position_by_doc_id[d0] == 0`` means stable ``d0`` is shown as
    ``D0``. Deterministic for a fixed seed.
    """
    ordered = list(example.documents)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    position_by_doc_id = {
        doc.doc_id: position for position, doc in enumerate(ordered)
    }
    return ordered, position_by_doc_id


def render_user_message(
    example: WorkerExample, ordered: List[WorkerDocument]
) -> str:
    """Render the question and document block shown to the worker."""
    blocks = [f"Question: {example.local_question}", "", "Documents:", ""]
    for position, doc in enumerate(ordered):
        blocks.append(f"D{position}: {doc.title}")
        blocks.append(doc.text)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def render_target(
    example: WorkerExample, position_by_doc_id: Dict[str, int]
) -> str:
    """Render the assistant target: evidence lines or NOT FOUND."""
    if not example.answerable:
        return NOT_FOUND
    lines = []
    for ev in example.evidence:
        position = position_by_doc_id[ev.doc_id]
        lines.append(f'D{position}: "{ev.sentence}"')
    return "\n".join(lines)


def build_messages(
    example: WorkerExample,
    instruction: str,
    shuffle_seed: int,
) -> Tuple[List[Dict[str, str]], str]:
    """Build chat messages and the matching target for one example.

    Returns ``(messages, target)`` where messages is system+user only
    (the assistant turn is the target, appended by the masking module).
    """
    ordered, position_by_doc_id = shuffle_and_remap(example, shuffle_seed)
    user = render_user_message(example, ordered)
    target = render_target(example, position_by_doc_id)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]
    return messages, target


def parse_output(text: str) -> Dict[str, Any]:
    """Parse a generated output into lines / NOT FOUND.

    Returns ``{abstained, lines: [{position, quote}], malformed}``.
    Only the format is judged here; content checks are separate.
    """
    stripped = text.strip()
    if stripped == NOT_FOUND:
        return {"abstained": True, "lines": [], "malformed": False}
    lines = []
    malformed = False
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TARGET_LINE_RE.match(line)
        if not match:
            malformed = True
            break
        lines.append({"position": int(match.group(1)), "quote": match.group(2)})
    if not lines and not malformed:
        malformed = True
    return {"abstained": False, "lines": lines, "malformed": malformed}


def evaluate_output(
    example: WorkerExample,
    ordered: List[WorkerDocument],
    output_text: str,
) -> Dict[str, Any]:
    """Evaluate a generated output against one example.

    Checks: parse/format, source validity, verbatim quote match,
    answer coverage (positives), abstention (negatives). Returns a dict
    of check results plus ``correct`` (coverage for positives,
    abstention for negatives — the same criterion used for validation
    selection).
    """
    parsed = parse_output(output_text)
    result: Dict[str, Any] = {
        "output": output_text,
        "parse_ok": not parsed["malformed"],
        "abstained": parsed["abstained"],
        "lines": parsed["lines"],
    }
    doc_texts = [doc.text for doc in ordered]
    doc_norm = [normalize_text(doc.text) for doc in ordered]

    if parsed["malformed"]:
        result.update(
            {
                "source_valid": False,
                "quotes_verbatim": False,
                "answer_covered": False,
                "abstention_correct": False,
                "correct": False,
            }
        )
        return result

    if parsed["abstained"]:
        result.update(
            {
                "source_valid": True,
                "quotes_verbatim": True,
                "answer_covered": False,
                "abstention_correct": not example.answerable,
                "correct": not example.answerable,
            }
        )
        return result

    # Source validity: every cited position exists.
    source_valid = all(
        0 <= line["position"] < len(ordered) for line in parsed["lines"]
    )
    # Verbatim quote match: the quoted text is a sentence of the cited
    # document (original text comparison; normalized recorded too).
    quotes_verbatim = True
    quote_details = []
    for line in parsed["lines"]:
        position = line["position"]
        quote = line["quote"]
        if not source_valid or position >= len(ordered):
            quote_details.append(
                {"position": position, "verbatim": False, "normalized_match": False}
            )
            quotes_verbatim = False
            continue
        verbatim = quote in doc_texts[position]
        normalized_match = (
            normalize_text(quote) in doc_norm[position] if quote else False
        )
        quote_details.append(
            {
                "position": position,
                "verbatim": verbatim,
                "normalized_match": normalized_match,
            }
        )
        if not verbatim:
            quotes_verbatim = False
    combined_quote = " ".join(line["quote"] for line in parsed["lines"])
    answer_covered = False
    if example.answerable:
        combined_norm = normalize_text(combined_quote)
        candidates = [example.answer] + list(example.answer_aliases)
        answer_covered = any(
            c and normalize_text(c) in combined_norm for c in candidates
        )
    correct = answer_covered if example.answerable else False
    result.update(
        {
            "source_valid": source_valid,
            "quotes_verbatim": quotes_verbatim,
            "quote_details": quote_details,
            "answer_covered": answer_covered,
            "abstention_correct": False,
            "correct": correct,
        }
    )
    return result

"""Minimal parser for Celab outputs (spec sec. 7).

Only three actions exist: ask Alice, ask Bob, final answer. An output is
actionable if and only if exactly one action type occurs and its marker
occurs exactly once:

- ``<TO>ALICE</TO>`` -> ask Alice
- ``<TO>BOB</TO>``   -> ask Bob
- ``<FINAL>...</FINAL>`` -> final answer (inner text is the prediction)

Anything else is a parse error. The scheduler must not guess the routing
target, rewrite the output, retry, or repair the format (spec sec. 7).
Markers are case-sensitive and incomplete markers do not match. Worker
outputs are never parsed; marker-looking text inside a worker reply has no
effect on routing or termination.

Design choices recorded here: a repeated marker of the same type is treated
as ambiguous (parse error); ``<FINAL></FINAL>`` is a *valid* final with an
empty prediction (evaluated as empty, distinct from a parse error, which
yields ``final_answer = null``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

TO_ALICE_RE = re.compile(r"<TO>ALICE</TO>")
TO_BOB_RE = re.compile(r"<TO>BOB</TO>")
FINAL_RE = re.compile(r"<FINAL>(.*?)</FINAL>", re.DOTALL)


@dataclass
class ParseResult:
    action: Optional[str]  # "ask_alice" | "ask_bob" | "final" | None
    body: Optional[str]  # inner <FINAL> text when action == "final"
    status: str  # "ok" | "error"
    error: Optional[str]


def parse_celab_output(text: str) -> ParseResult:
    """Parse one Celab output into exactly one action, or a parse error."""
    final_bodies = FINAL_RE.findall(text)
    counts = {
        "ask_alice": len(TO_ALICE_RE.findall(text)),
        "ask_bob": len(TO_BOB_RE.findall(text)),
        "final": len(final_bodies),
    }
    present = [(action, count) for action, count in counts.items() if count > 0]
    if not present:
        return ParseResult(
            None,
            None,
            "error",
            "no action marker found (expected <TO>ALICE</TO>, <TO>BOB</TO>, "
            "or <FINAL>...</FINAL>)",
        )
    if len(present) > 1:
        return ParseResult(
            None,
            None,
            "error",
            "multiple action types present in one output: "
            + ", ".join(f"{action!r} x{count}" for action, count in present),
        )
    action, count = present[0]
    if count > 1:
        return ParseResult(
            None,
            None,
            "error",
            f"action marker {action!r} occurs {count} times (ambiguous)",
        )
    if action == "final":
        body = final_bodies[0].strip()
        return ParseResult("final", body, "ok", None)
    return ParseResult(action, None, "ok", None)


def extract_final_answer(text: str) -> ParseResult:
    """Final-answer extraction for the forced-final output.

    Only a uniquely parseable ``<FINAL>...</FINAL>`` counts. A ``<TO>``
    marker (or anything else) in the forced-final output is a failure and
    yields ``final_answer = null`` (spec sec. 9.3).
    """
    result = parse_celab_output(text)
    if result.action == "final":
        return result
    return ParseResult(
        None,
        None,
        "error",
        result.error or "output does not contain a unique <FINAL>...</FINAL>",
    )

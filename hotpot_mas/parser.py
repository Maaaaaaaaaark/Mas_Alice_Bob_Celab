"""Minimal parser for Celab outputs (spec sec. 7).

Only three actions exist: ask Alice, ask Bob, final answer:

- ``<TO>ALICE</TO>`` -> ask Alice
- ``<TO>BOB</TO>``   -> ask Bob
- ``<FINAL>...</FINAL>`` -> final answer (inner text is the prediction)

The canonical forms above remain the normal path. Two narrowly scoped,
audited compatibility fallbacks cover formatting failures observed in the
v4 pilot without inferring an action from natural language:

- a uniquely occurring, fully closed route marker may differ only in case;
- a unique, fully closed ``<FINAL>`` that occurs after every route marker
  wins over those earlier route markers (the model emitted an entire action
  sequence in one generation and ended it with an explicit final answer).

Incomplete route markers are deliberately not accepted. Worker outputs are
never parsed; marker-looking text inside a worker reply has no effect on
routing or termination.

Design choices recorded here: a repeated route marker without a later final
is ambiguous; ``<FINAL></FINAL>`` is a *valid* final with an empty prediction.
Exactly one unclosed ``<FINAL>`` marker, with no routing/closing marker, is
also accepted and treats the remainder as the answer. Every compatibility
fallback has a distinct parse status so it is never silently presented as a
canonical parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ROUTE_CASEFOLD_RE = re.compile(r"<TO>(ALICE|BOB)</TO>", re.IGNORECASE)
FINAL_RE = re.compile(r"<FINAL>(.*?)</FINAL>", re.DOTALL)
FINAL_OPEN = "<FINAL>"
FINAL_CLOSE = "</FINAL>"


@dataclass
class ParseResult:
    action: Optional[str]  # "ask_alice" | "ask_bob" | "final" | None
    body: Optional[str]  # inner <FINAL> text when action == "final"
    status: str  # "ok" | an audited ``ok_*_fallback`` | "error"
    error: Optional[str]


def parse_celab_output(text: str) -> ParseResult:
    """Parse one Celab output into exactly one action, or a parse error."""
    final_bodies = FINAL_RE.findall(text)
    final_open_count = text.count(FINAL_OPEN)
    final_close_count = text.count(FINAL_CLOSE)
    route_matches = list(ROUTE_CASEFOLD_RE.finditer(text))

    # A fully closed final at the end of an emitted action sequence is an
    # unambiguous terminal action. This recovers outputs such as
    # ``<TO>ALICE</TO> ... <TO>BOB</TO> ... <FINAL>x</FINAL>`` while still
    # rejecting a route marker that appears after the final.
    if (
        final_open_count == 1
        and final_close_count == 1
        and len(final_bodies) == 1
    ):
        final_start = text.index(FINAL_OPEN)
        if not route_matches:
            return ParseResult("final", final_bodies[0].strip(), "ok", None)
        if all(match.end() <= final_start for match in route_matches):
            return ParseResult(
                "final",
                final_bodies[0].strip(),
                "ok_terminal_final_precedence_fallback",
                None,
            )
        return ParseResult(
            None,
            None,
            "error",
            "route marker occurs after <FINAL> marker (ambiguous)",
        )

    if final_open_count:
        if (
            final_open_count == 1
            and final_close_count == 0
            and not route_matches
        ):
            body = text.split(FINAL_OPEN, 1)[1].strip()
            return ParseResult(
                "final", body, "ok_unclosed_final_fallback", None
            )
        return ParseResult(
            None,
            None,
            "error",
            "malformed or ambiguous <FINAL> structure",
        )

    if not route_matches:
        return ParseResult(
            None,
            None,
            "error",
            "no action marker found (expected <TO>ALICE</TO>, <TO>BOB</TO>, "
            "or <FINAL>...</FINAL>)",
        )
    if len(route_matches) > 1:
        return ParseResult(
            None,
            None,
            "error",
            f"route marker occurs {len(route_matches)} times (ambiguous)",
        )

    match = route_matches[0]
    target = match.group(1).lower()
    action = f"ask_{target}"
    canonical = f"<TO>{target.upper()}</TO>"
    status = "ok" if match.group(0) == canonical else "ok_casefold_route_fallback"
    return ParseResult(action, None, status, None)


def extract_final_answer(text: str) -> ParseResult:
    """Final-answer extraction for the forced-final output.

    A unique closed final, the explicitly logged unclosed-final fallback, or
    a terminal-final-precedence fallback counts. Other output yields
    ``final_answer = null``.
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

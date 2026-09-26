"""Message and event records (spec sec. 15.3).

A ``Message`` is one turn in an agent's visible history. Chat-template roles
are derived per viewer (own messages become assistant turns, partner
messages become user turns), so one ``Message`` object can safely live in
two agents' histories without sharing any other state between those agents.

An ``Event`` is the logged trace entry for every model generation and every
controller action. Controller events have ``generated_tokens = 0``; the
forced-final controller instruction is such an event, and its text is also
stored as a ``Message`` so it becomes part of Celab's next input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    message_id: str
    speaker: str  # "alice" | "bob" | "celab" | "controller"
    recipient: Optional[str]
    content: str


@dataclass
class Event:
    event_index: int
    event_type: str  # "model_generation" | "controller"
    decision_step: Optional[int]
    speaker: str
    recipient: Optional[str] = None
    message_id: Optional[str] = None
    raw_output: Optional[str] = None
    parsed_action: Optional[str] = None  # "ask_alice" | "ask_bob" | "final" | None
    parsed_body: Optional[str] = None
    parse_status: str = "not_parsed"  # "ok" | "error" | "not_parsed"
    parse_error: Optional[str] = None
    input_tokens: int = 0
    generated_tokens: int = 0
    generation_seed: Optional[int] = None
    finish_reason: Optional[str] = None  # "eos" | "length" (model generations)
    generation_cap_reached: bool = False
    forced_final: bool = False
    controller_subtype: Optional[str] = None
    visible_history_message_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_index": self.event_index,
            "event_type": self.event_type,
            "decision_step": self.decision_step,
            "speaker": self.speaker,
            "recipient": self.recipient,
            "message_id": self.message_id,
            "raw_output": self.raw_output,
            "parsed_action": self.parsed_action,
            "parsed_body": self.parsed_body,
            "parse_status": self.parse_status,
            "parse_error": self.parse_error,
            "input_tokens": self.input_tokens,
            "generated_tokens": self.generated_tokens,
            "generation_seed": self.generation_seed,
            "finish_reason": self.finish_reason,
            "generation_cap_reached": self.generation_cap_reached,
            "forced_final": self.forced_final,
            "controller_subtype": self.controller_subtype,
            "visible_history_message_ids": list(self.visible_history_message_ids),
        }

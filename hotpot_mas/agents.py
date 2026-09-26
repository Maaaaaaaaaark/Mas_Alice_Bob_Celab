"""The three logical agents (spec sec. 6).

Alice and Bob each hold one private evidence document; Celab holds the
question. All three are independent objects with fully separate histories;
they share only the frozen engine. A message that crosses a link is stored
in both sender and receiver histories (relaying through Celab, spec sec. 3),
and chat-template roles are derived per viewer, so a shared ``Message``
object never leaks any other agent state.

``check_agent_isolation`` inspects ONLY the rendered system prompt: message
contents are explicit communication and must never be treated as leakage.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

from .messages import Message


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        engine: Any,
        run_seed: int,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.engine = engine
        self.run_seed = run_seed
        self.history: List[Message] = []
        self._ids = itertools.count()

    def next_message_id(self) -> str:
        """Return a message id unique within this run (global counter)."""
        # The counter is per-agent but only message CREATION uses it; ids are
        # made globally unique by prefixing the agent name and an index.
        return f"{self.name}-m{next(self._ids)}"

    def add_message(self, message: Message) -> None:
        """Append an existing message to this agent's visible history."""
        self.history.append(message)

    def visible_message_ids(self) -> List[str]:
        return [message.message_id for message in self.history]

    def build_chat_messages(self) -> List[Dict[str, str]]:
        """Chat-template input: system prompt + history (own=assistant).

        The Gemma-3 template accepts a system role only as messages[0] and
        folds it into the first user turn; roles must start with user and
        strictly alternate. Worker calls always start with a user request
        and Celab's first call with the controller question turn, so the
        sequence is valid. A forced-final instruction is appended as a user
        turn by the orchestrator.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for message in self.history:
            role = "assistant" if message.speaker == self.name else "user"
            messages.append({"role": role, "content": message.content})
        return messages

    def generate(self) -> Any:
        """Call the shared engine with this agent's current visible input."""
        return self.engine.generate(
            self.build_chat_messages(), self.run_seed, self.name
        )


def make_agents(
    prompts: Any,
    engine: Any,
    run_seed: int,
    evidence_alice: str,
    evidence_bob: str,
) -> Dict[str, Agent]:
    """Create the three independent agents for one run."""
    alice = Agent(
        "alice",
        prompts.render("alice", private_evidence=evidence_alice),
        engine,
        run_seed,
    )
    bob = Agent(
        "bob",
        prompts.render("bob", private_evidence=evidence_bob),
        engine,
        run_seed,
    )
    celab = Agent("celab", prompts.render("celab"), engine, run_seed)
    return {"alice": alice, "bob": bob, "celab": celab}


def check_agent_isolation(
    agent: Agent, forbidden_system_content: List[str]
) -> List[str]:
    """Return the forbidden substrings present in the agent's system prompt.

    Only the system prompt is checked: history message contents are explicit
    inter-agent communication (relayed through Celab) and must never be
    flagged as leakage (spec sec. 18.5).
    """
    return [
        text for text in forbidden_system_content if text in agent.system_prompt
    ]

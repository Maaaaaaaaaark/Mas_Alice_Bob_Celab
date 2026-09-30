"""The three logical agents (spec sec. 6).

Alice and Bob each hold private evidence; C holds the question. All three are
independent objects with fully separate histories. Depending on the versioned
experiment config, they either share one frozen engine or use three separately
loaded frozen engines. A message that crosses a link is stored
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
from .seeds import derive_generation_seed


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
        self._generation_calls = itertools.count()

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
            # Gemma requires strict user/assistant alternation.  The forced
            # final controller instruction follows a worker reply, so both
            # are user-visible turns; preserve both texts while coalescing
            # adjacent same-role inputs into one chat-template turn.
            if len(messages) > 1 and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + message.content
            else:
                messages.append({"role": role, "content": message.content})
        return messages

    def generate(self) -> Any:
        """Call this agent's assigned engine with its current visible input."""
        call_index = next(self._generation_calls)
        generation_seed = derive_generation_seed(
            self.run_seed, self.name, call_index
        )
        return self.engine.generate(
            self.build_chat_messages(), generation_seed, self.name
        )


def make_agents(
    prompts: Any,
    engine: Any,
    run_seed: int,
    evidence_alice: str,
    evidence_bob: str,
    question: str = "",
) -> Dict[str, Agent]:
    """Create agents with either one shared engine or per-agent engines."""
    if isinstance(engine, dict):
        required = {"alice", "bob", "celab"}
        missing = required - set(engine)
        if missing:
            raise ValueError(f"missing engines for agents: {sorted(missing)}")
        engines = engine
    else:
        engines = {name: engine for name in ("alice", "bob", "celab")}
    alice = Agent(
        "alice",
        prompts.render(
            "alice", private_evidence=evidence_alice, question=question
        ),
        engines["alice"],
        run_seed,
    )
    bob = Agent(
        "bob",
        prompts.render(
            "bob", private_evidence=evidence_bob, question=question
        ),
        engines["bob"],
        run_seed,
    )
    celab = Agent(
        "celab", prompts.render("celab"), engines["celab"], run_seed
    )
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

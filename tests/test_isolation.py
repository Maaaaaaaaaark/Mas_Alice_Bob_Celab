"""Information-isolation tests (spec sec. 18.1-18.5, 18.16).

End-to-end checks run real trajectories against the MockEngine and inspect
every model input the engine received, so isolation is verified exactly as
it holds in production code paths (message relaying is not flagged as
leakage, per spec sec. 18.5).
"""

from __future__ import annotations

from typing import Any, Dict, List

from hotpot_mas.agents import Agent, check_agent_isolation

from conftest import E_A_MARKER, E_B_MARKER, SAMPLE_QUESTION

# Scripts where Alice's reply quotes her own evidence (an explicit relay
# through Celab, which must NOT be treated as leakage).
RELAY_SCRIPTS: Dict[str, List[str]] = {
    "celab": [
        "Celab: <TO>ALICE</TO> Tell me about Alpha City.",
        "Celab: <TO>BOB</TO> Tell me about Beta City.",
        "Celab: <FINAL>The Connector Bridge</FINAL>",
    ],
    "alice": [f"Alice: my evidence says {E_A_MARKER}."],
    "bob": [f"Bob: my evidence says {E_B_MARKER}."],
}


def _flatten_input(call: Any) -> str:
    """All text the model saw in one call (system + history contents)."""
    return "\n".join(message["content"] for message in call.messages)


def test_alice_input_never_contains_bob_evidence(run_with_mock):
    record, engine = run_with_mock(scripts=dict(RELAY_SCRIPTS))
    assert record["termination_reason"] == "natural_final"
    assert len(engine.calls_by_speaker["alice"]) == 1
    for call in engine.calls_by_speaker["alice"]:
        assert E_B_MARKER not in _flatten_input(call)
        # Her own evidence IS visible to her (system prompt only).
        assert E_A_MARKER in _flatten_input(call)


def test_bob_input_never_contains_alice_evidence(run_with_mock):
    record, engine = run_with_mock(scripts=dict(RELAY_SCRIPTS))
    assert record["termination_reason"] == "natural_final"
    for call in engine.calls_by_speaker["bob"]:
        assert E_A_MARKER not in _flatten_input(call)
        assert E_B_MARKER in _flatten_input(call)


def test_worker_initial_input_has_no_question(run_with_mock):
    record, engine = run_with_mock(scripts=dict(RELAY_SCRIPTS))
    question_text = SAMPLE_QUESTION["question"]
    for speaker in ("alice", "bob"):
        calls = engine.calls_by_speaker[speaker]
        assert calls, f"no calls recorded for {speaker}"
        system = calls[0].messages[0]["content"]
        assert question_text not in system
        # The question is also absent from the first user turn (the request).
        assert question_text not in _flatten_input(calls[0])


def test_celab_initial_input_has_no_evidence(run_with_mock):
    record, engine = run_with_mock(scripts=dict(RELAY_SCRIPTS))
    first_celab_input = _flatten_input(engine.calls_by_speaker["celab"][0])
    assert E_A_MARKER not in first_celab_input
    assert E_B_MARKER not in first_celab_input
    # The question is there (first user turn).
    assert SAMPLE_QUESTION["question"] in first_celab_input


def test_relay_through_celab_is_explicit_communication_not_leakage(
    run_with_mock,
):
    record, engine = run_with_mock(scripts=dict(RELAY_SCRIPTS))
    # After Alice's reply, Celab legitimately sees the quoted marker.
    second_celab_input = _flatten_input(engine.calls_by_speaker["celab"][1])
    assert E_A_MARKER in second_celab_input
    # Alice never sees Bob's marker even after the relay.
    for call in engine.calls_by_speaker["alice"]:
        assert E_B_MARKER not in _flatten_input(call)


def test_check_agent_isolation_checks_only_system_prompt(run_with_mock):
    record, engine = run_with_mock(scripts=dict(RELAY_SCRIPTS))
    assert record["termination_reason"] == "natural_final"

    # Rebuild Celab with the full history he accumulated during the run.
    celab_inputs = engine.calls_by_speaker["celab"][-1].messages
    system_prompt = celab_inputs[0]["content"]
    celab = Agent("celab", system_prompt, engine, 0)
    # Alice's reply quoted E_A_MARKER; that text reaches Celab as explicit
    # communication. check_agent_isolation must not flag it, because it
    # inspects ONLY the system prompt.
    assert check_agent_isolation(celab, [E_A_MARKER, E_B_MARKER]) == []
    # If the marker leaked INTO the system prompt, it would be flagged.
    leaked = Agent("celab", system_prompt + f" {E_A_MARKER}", engine, 0)
    assert check_agent_isolation(leaked, [E_A_MARKER]) == [E_A_MARKER]


def test_histories_are_fully_independent(run_with_mock):
    record, engine = run_with_mock(scripts=dict(RELAY_SCRIPTS))
    alice_all = "\n".join(
        _flatten_input(call) for call in engine.calls_by_speaker["alice"]
    )
    bob_all = "\n".join(
        _flatten_input(call) for call in engine.calls_by_speaker["bob"]
    )
    # Neither worker ever sees the other worker's own messages.
    assert "Bob:" not in alice_all
    assert "Alice:" not in bob_all

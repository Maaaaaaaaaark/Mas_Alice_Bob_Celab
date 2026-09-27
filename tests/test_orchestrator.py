"""Orchestrator state-machine tests (spec sec. 18.6-18.11, 18.15).

All runs use the MockEngine, so no GPU is needed. Token counts come from
the mock's whitespace splitter and are self-consistent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from hotpot_mas.model_engine import MockEngine
from hotpot_mas.orchestrator import Orchestrator
from hotpot_mas.prompts import CENTRALIZED_PROMPT_NAMES, PromptSet


def _model_events(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e for e in record["events"] if e["event_type"] == "model_generation"]


def _controller_events(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e for e in record["events"] if e["event_type"] == "controller"]


def _ask_scripts(steps: int, target: str) -> Dict[str, List[str]]:
    return {
        "celab": [f"Celab: <TO>{target.upper()}</TO> query {i}" for i in range(steps)],
        target: [f"{target.title()}: reply {i}" for i in range(steps)],
    }


def test_natural_run_counts_decision_steps(run_with_mock):
    # 3 Celab calls (ask, ask, final), 2 worker replies.
    record, engine = run_with_mock()
    assert record["termination_reason"] == "natural_final"
    assert record["natural_termination"] is True
    assert record["decision_steps"] == 3
    assert record["num_alice_queries"] == 1
    assert record["num_bob_queries"] == 1
    assert record["num_alice_responses"] == 1
    assert record["num_bob_responses"] == 1
    assert record["num_messages"] == 5
    assert record["final_answer"] == "The Connector Bridge"
    assert record["f1"] == 1.0 and record["em"] == 1.0
    # Worker events carry the step that triggered them and do NOT add steps.
    steps = [e["decision_step"] for e in _model_events(record)]
    assert steps == [1, 1, 2, 2, 3]


def test_natural_unclosed_final_uses_audited_fallback(run_with_mock):
    record, _ = run_with_mock(
        scripts={"celab": ["Celab: <FINAL>The Connector Bridge"]}
    )
    assert record["termination_reason"] == "natural_final"
    assert record["natural_termination"] is True
    assert record["final_answer"] == "The Connector Bridge"
    assert record["parse_status"] == "ok_unclosed_final_fallback"
    assert record["final_parse_fallback"] is True
    assert record["em"] == 1.0
    final_event = _model_events(record)[0]
    assert final_event["parse_status"] == "ok_unclosed_final_fallback"


def test_casefold_route_fallback_is_audited_and_run_continues(run_with_mock):
    scripts = {
        "celab": [
            "Celab: <TO>Alice</TO> q1",
            "Celab: <FINAL>The Connector Bridge</FINAL>",
        ],
        "alice": ["Alice: reply"],
    }
    record, _ = run_with_mock(scripts=scripts)
    assert record["termination_reason"] == "natural_final"
    assert record["num_alice_queries"] == 1
    assert record["protocol_parse_fallback"] is True
    assert record["casefold_route_fallback"] is True
    assert record["terminal_final_precedence_fallback"] is False
    assert record["parse_fallback_events"] == 1
    assert record["parse_fallback_statuses"] == [
        "ok_casefold_route_fallback"
    ]
    assert _model_events(record)[0]["parse_status"] == (
        "ok_casefold_route_fallback"
    )


def test_terminal_final_precedence_fallback_is_scored_officially(run_with_mock):
    record, _ = run_with_mock(
        scripts={
            "celab": [
                "Celab: <TO>ALICE</TO> x <TO>BOB</TO> y "
                "<FINAL>The Connector Bridge</FINAL>"
            ]
        }
    )
    assert record["termination_reason"] == "natural_final"
    assert record["final_answer"] == "The Connector Bridge"
    assert record["em"] == 1.0
    assert record["protocol_parse_fallback"] is True
    assert record["terminal_final_precedence_fallback"] is True
    assert record["casefold_route_fallback"] is False
    assert record["parse_fallback_events"] == 1


def test_worker_replies_do_not_increment_decision_steps(run_with_mock):
    scripts = {
        "celab": ["Celab: <TO>ALICE</TO> q1", "Celab: <TO>ALICE</TO> q2",
                  "Celab: <TO>ALICE</TO> q3", "Celab: <FINAL>X</FINAL>"],
        "alice": ["Alice: r1", "Alice: r2", "Alice: r3"],
    }
    record, engine = run_with_mock(scripts=scripts)
    # 4 Celab calls but only 4 steps; the 3 worker replies add nothing.
    assert record["decision_steps"] == 4
    assert record["num_alice_queries"] == 3
    assert record["num_messages"] == 7


def test_final_call_counts_as_decision_step(run_with_mock):
    record, engine = run_with_mock()
    final_event = _model_events(record)[-1]
    assert final_event["speaker"] == "celab"
    assert final_event["parsed_action"] == "final"
    assert final_event["decision_step"] == 3


def test_forced_final_after_cap_keeps_steps_at_20(run_with_mock):
    scripts = _ask_scripts(20, "alice")
    # The 21st Celab script is consumed by the forced-final call.
    scripts["celab"].append("Celab: <FINAL>The Connector Bridge</FINAL>")
    record, engine = run_with_mock(scripts=scripts)

    assert record["decision_steps"] == 20
    assert record["cap_reached"] is True
    assert record["decision_cap_reached"] is True
    assert record["forced_final_calls"] == 1
    assert record["termination_reason"] == "forced_final"
    assert record["natural_termination"] is False
    assert record["final_answer"] == "The Connector Bridge"

    # No event ever carries decision_step 21.
    assert all(e["decision_step"] is None or e["decision_step"] <= 20
               for e in record["events"])
    # The forced-final generation event is marked and keeps step 20.
    forced = [e for e in _model_events(record) if e["forced_final"]]
    assert len(forced) == 1
    assert forced[0]["decision_step"] == 20
    # Two cap-related controller events, both with zero generated tokens.
    subtypes = [e["controller_subtype"] for e in _controller_events(record)]
    assert "decision_cap_reached" in subtypes
    assert "forced_final_instruction" in subtypes
    assert all(e["generated_tokens"] == 0 for e in _controller_events(record))
    # The real Gemma template requires strict role alternation.  The worker
    # reply and controller instruction are both user-visible, so Agent must
    # coalesce them without dropping either text.
    forced_call = engine.calls_by_speaker["celab"][-1]
    roles = [message["role"] for message in forced_call.messages[1:]]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    assert "maximum number of interaction steps" in forced_call.messages[-1]["content"]


def test_forced_final_tokens_are_counted(run_with_mock):
    scripts = _ask_scripts(20, "alice")
    scripts["celab"].append("Celab: <FINAL>The Connector Bridge</FINAL>")
    record, engine = run_with_mock(scripts=scripts)
    forced = [e for e in _model_events(record) if e["forced_final"]][0]
    assert forced["generated_tokens"] > 0
    assert forced["generated_tokens"] == len(
        "Celab: <FINAL>The Connector Bridge</FINAL>".split()
    )
    # Included in the run-level total (sum over all model events).
    event_sum = sum(e["generated_tokens"] for e in _model_events(record))
    assert record["total_generated_tokens"] == event_sum


def test_forced_final_parse_failure_gives_null_answer(run_with_mock):
    scripts = _ask_scripts(20, "alice")
    scripts["celab"].append("Celab: I give up, no format")
    record, engine = run_with_mock(scripts=scripts)
    assert record["termination_reason"] == "forced_final_parse_failure"
    assert record["final_answer"] is None
    assert record["parse_status"] == "error"
    assert record["f1"] == 0.0 and record["em"] == 0.0
    assert record["cap_reached"] is True
    # The unparseable forced output is still counted in tokens.
    forced = [e for e in _model_events(record) if e["forced_final"]][0]
    assert forced["generated_tokens"] > 0


def test_forced_unclosed_final_uses_audited_fallback(run_with_mock):
    scripts = _ask_scripts(20, "alice")
    scripts["celab"].append("Celab: <FINAL>The Connector Bridge")
    record, _ = run_with_mock(scripts=scripts)
    assert record["termination_reason"] == "forced_final"
    assert record["natural_termination"] is False
    assert record["final_answer"] == "The Connector Bridge"
    assert record["parse_status"] == "ok_unclosed_final_fallback"
    assert record["final_parse_fallback"] is True
    assert record["em"] == 1.0
    forced = [e for e in _model_events(record) if e["forced_final"]]
    assert forced[0]["parse_status"] == "ok_unclosed_final_fallback"


def test_parse_error_ends_run_without_retry_or_repair(run_with_mock):
    record, engine = run_with_mock(
        scripts={"celab": ["Celab: no markers here at all"]}
    )
    assert record["termination_reason"] == "parse_error"
    assert record["parse_status"] == "error"
    assert record["parse_error"] is not None
    assert record["final_answer"] is None
    assert record["f1"] == 0.0 and record["em"] == 0.0
    assert record["decision_steps"] == 1
    # Exactly one model call happened: no guessing, no retry, no forced final.
    assert len(engine.calls) == 1
    assert engine.calls[0].speaker == "celab"
    assert record["num_messages"] == 1
    # Raw output is preserved verbatim in the event.
    event = _model_events(record)[0]
    assert event["raw_output"] == "Celab: no markers here at all"
    assert event["parse_status"] == "error"
    # Controller notice event closes the run.
    subtypes = [e["controller_subtype"] for e in _controller_events(record)]
    assert "parse_error_termination" in subtypes


def test_generation_cap_is_separate_from_decision_cap(run_with_mock):
    # Bob's reply hits the generation cap; the run itself still terminates
    # naturally with no decision cap.
    record, engine = run_with_mock(cap_speakers=["bob"])
    assert record["termination_reason"] == "natural_final"
    assert record["cap_reached"] is False
    assert record["decision_cap_reached"] is False
    assert record["generation_cap_reached"] is True
    assert record["generation_cap_agents"] == ["bob"]
    assert record["generation_cap_events"] == 1
    bob_event = [
        e for e in _model_events(record) if e["speaker"] == "bob"
    ][0]
    assert bob_event["generation_cap_reached"] is True
    assert bob_event["finish_reason"] == "length"
    # Cap runs still end normally, not via error.
    assert record["termination_reason"] == "natural_final"


def test_worker_final_marker_does_not_terminate_run(run_with_mock):
    scripts = {
        "celab": [
            "Celab: <TO>ALICE</TO> tell me about Alpha City.",
            "Celab: <FINAL>The Connector Bridge</FINAL>",
        ],
        "alice": ["Alice: sure, <FINAL>whatever</FINAL> is just text."],
    }
    record, engine = run_with_mock(scripts=scripts)
    # The worker reply is never parsed, so the run reaches Celab's real final.
    assert record["termination_reason"] == "natural_final"
    assert record["decision_steps"] == 2
    assert record["final_answer"] == "The Connector Bridge"


def test_total_generated_tokens_equals_event_sum(run_with_mock):
    record, engine = run_with_mock()
    event_sum = sum(e["generated_tokens"] for e in _model_events(record))
    assert record["total_generated_tokens"] == event_sum
    assert record["total_generated_tokens"] > 0
    # Per-agent breakdown sums correctly too.
    by_agent: Dict[str, int] = {}
    for e in _model_events(record):
        by_agent[e["speaker"]] = by_agent.get(e["speaker"], 0) + e[
            "generated_tokens"
        ]
    assert record["generated_tokens_alice"] == by_agent.get("alice", 0)
    assert record["generated_tokens_bob"] == by_agent.get("bob", 0)
    assert record["generated_tokens_celab"] == by_agent.get("celab", 0)


def test_each_agent_call_has_independent_reproducible_seed(run_with_mock):
    record, engine = run_with_mock(run_seed=7)
    seeds = [call.seed for call in engine.calls]
    assert len(seeds) == len(set(seeds))
    assert [e["generation_seed"] for e in _model_events(record)] == seeds

    _, engine2 = run_with_mock(run_seed=7)
    assert [call.seed for call in engine2.calls] == seeds


def test_engine_exception_is_recorded_as_error_termination(run_with_mock):
    # Alice's scripted queue is empty -> her generate() raises mid-run.
    scripts = {
        "celab": ["Celab: <TO>ALICE</TO> q1"],
        "alice": [],
    }
    record, engine = run_with_mock(scripts=scripts)
    assert record["termination_reason"] == "error"
    assert record["final_answer"] is None
    # Partial record still contains the events up to the failure.
    assert len(_model_events(record)) == 1
    subtypes = [e["controller_subtype"] for e in _controller_events(record)]
    assert "error_termination" in subtypes


def test_controller_events_have_zero_generated_tokens(run_with_mock):
    record, engine = run_with_mock()
    for event in _controller_events(record):
        assert event["generated_tokens"] == 0
        assert event["input_tokens"] == 0
    # initial_task is the first event and carries no step.
    first = record["events"][0]
    assert first["controller_subtype"] == "initial_task"
    assert first["decision_step"] is None


def test_evaluation_uses_only_final_inner_content(run_with_mock):
    # Text OUTSIDE the <FINAL> marker is ignored for scoring; only the
    # inner text is the prediction. The gold answer also appears in Alice's
    # reply, but worker replies never influence scoring.
    scripts = {
        "celab": [
            "Celab: <TO>ALICE</TO> q1",
            "Celab: <FINAL>The Connector Bridge</FINAL> plus stray extra text",
        ],
        "alice": ["Alice: The gold answer is The Connector Bridge."],
    }
    record, engine = run_with_mock(scripts=scripts)
    assert record["termination_reason"] == "natural_final"
    assert record["final_answer"] == "The Connector Bridge"
    assert record["em"] == 1.0
    assert record["f1"] == 1.0

    # Text INSIDE the marker is part of the prediction and is scored as-is.
    scripts["celab"][1] = (
        "Celab: <FINAL>The Connector Bridge plus extra</FINAL>"
    )
    record2, _ = run_with_mock(scripts=scripts)
    assert record2["final_answer"] == "The Connector Bridge plus extra"
    assert record2["em"] == 0.0
    assert record2["f1"] < 1.0


def test_worker_clarification_round_trip_is_logged(run_with_mock):
    scripts = {
        "celab": [
            "Celab: <TO>ALICE</TO> When did it end?",
            "Celab: <TO>ALICE</TO> I mean the entire war.",
            "Celab: <FINAL>The Connector Bridge</FINAL>",
        ],
        "alice": [
            "Alice: CLARIFY: Which event does 'it' refer to?",
            "Alice: The requested event ended in 1922.",
        ],
    }
    record, _ = run_with_mock(
        scripts=scripts, clarification_enabled=True
    )
    assert record["num_alice_clarification_requests"] == 1
    assert record["num_bob_clarification_requests"] == 0
    assert record["num_clarification_requests"] == 1
    assert record["num_clarification_round_trips_completed"] == 1
    assert record["clarification_protocol_violations"] == 0
    assert record["clarification_triggered"] is True
    alice_events = [
        event
        for event in _model_events(record)
        if event["speaker"] == "alice"
    ]
    assert alice_events[0]["parsed_action"] == "clarify"
    assert alice_events[0]["parse_status"] == "worker_clarification"


def test_clarification_wrong_next_target_is_observed_not_rewritten(
    run_with_mock,
):
    scripts = {
        "celab": [
            "Celab: <TO>ALICE</TO> When did it end?",
            "Celab: <TO>BOB</TO> Tell me the answer.",
            "Celab: <FINAL>The Connector Bridge</FINAL>",
        ],
        "alice": ["Alice: CLARIFY: Which event?"],
        "bob": ["Bob: The Connector Bridge."],
    }
    record, _ = run_with_mock(
        scripts=scripts, clarification_enabled=True
    )
    assert record["termination_reason"] == "natural_final"
    assert record["num_bob_queries"] == 1
    assert record["num_clarification_round_trips_completed"] == 0
    assert record["clarification_protocol_violations"] == 1


def test_worker_clarification_limit_is_nonfatal(run_with_mock):
    scripts = {
        "celab": [
            "Celab: <TO>ALICE</TO> q1",
            "Celab: <TO>ALICE</TO> q2",
            "Celab: <FINAL>The Connector Bridge</FINAL>",
        ],
        "alice": [
            "Alice: CLARIFY: first question?",
            "Alice: CLARIFY: second question?",
        ],
    }
    record, _ = run_with_mock(
        scripts=scripts,
        clarification_enabled=True,
        clarification_max_per_worker=1,
    )
    assert record["num_clarification_requests"] == 1
    assert record["ignored_clarification_requests"] == 1
    assert record["termination_reason"] == "natural_final"


def test_clarify_text_is_plain_worker_reply_when_feature_disabled(
    run_with_mock,
):
    scripts = {
        "celab": [
            "Celab: <TO>ALICE</TO> q1",
            "Celab: <FINAL>The Connector Bridge</FINAL>",
        ],
        "alice": ["Alice: CLARIFY: Which event?"],
    }
    record, _ = run_with_mock(scripts=scripts)
    assert record["num_clarification_requests"] == 0
    alice_event = [
        event
        for event in _model_events(record)
        if event["speaker"] == "alice"
    ][0]
    assert alice_event["parse_status"] == "not_parsed"


def test_centralized_reader_sees_question_and_both_evidence(
    base_config, sample_question
):
    repo_root = Path(__file__).resolve().parent.parent
    prompts = PromptSet(
        repo_root / "prompts_centralized", names=CENTRALIZED_PROMPT_NAMES
    )
    cfg = replace(base_config, architecture="centralized_reader")
    engine = MockEngine(scripts={"celab": ["The Connector Bridge"]})
    orchestrator = Orchestrator(
        cfg, prompts, engine, {"environment": "mock-test"}
    )
    record = orchestrator.run_one(
        sample_question, 0, 0, "test_q0001-seed-0"
    )

    assert record["architecture"] == "centralized_reader"
    assert record["termination_reason"] == "direct_answer"
    assert record["final_answer"] == "The Connector Bridge"
    assert record["f1"] == 1.0 and record["em"] == 1.0
    assert record["decision_steps"] == 1
    assert record["num_messages"] == 1
    assert record["total_generated_tokens"] > 0
    user_text = engine.calls[0].messages[-1]["content"]
    assert sample_question["question"] in user_text
    assert sample_question["evidence_alice"] in user_text
    assert sample_question["evidence_bob"] in user_text

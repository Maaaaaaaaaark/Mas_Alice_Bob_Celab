"""Single-run orchestration (spec sec. 7-9, 15).

State machine for one (question, run_seed) trajectory:

1. seed_all(run_seed); create the three independent agents; seed the
   question into Celab's history as a controller "initial_task" message
   (Celab's first user turn, so the Gemma-3 template keeps the system
   prompt).
2. While ``decision_steps < max_decision_steps``: call Celab (each call is
   one decision step, including the final call and any parse-error call),
   parse the output, and either
   - final  -> natural termination, evaluate, stop;
   - ask_*  -> deliver the request to the worker, call the worker (worker
     replies do NOT increment decision steps), continue;
   - parse error -> record it, terminate with final_answer=None, stop.
3. If the loop ends without termination: decision cap reached; append the
   fixed forced-final instruction (controller message + controller event,
   generated_tokens=0), call Celab once more with forced_final=True (this
   does NOT create a step 21; its tokens ARE counted), and extract the
   answer or record forced_final_parse_failure.

The parser may apply only its explicitly logged protocol compatibility
fallbacks; natural-language actions are never guessed, rewritten, or
retried. In the optional clarification condition, worker outputs are only
classified when they begin with the explicit ``CLARIFY:`` action. Other
worker text, including marker-looking text, remains free-form and cannot
terminate the run.

Token accounting (spec sec. 12): ``generated_tokens`` of each event comes
from the engine (real tokenizer for HFEngine); ``total_generated_tokens``
is the sum over alice + bob + celab events, including parse-error outputs
and the forced-final call. Input tokens are auxiliary.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .agents import Agent, make_agents
from .evaluation import evaluate_answer
from .messages import Event, Message
from .parser import extract_final_answer, parse_celab_output, parse_worker_reply
from .seeds import seed_all


class Orchestrator:
    """Runs one trajectory and returns the complete run record dict."""

    def __init__(
        self,
        cfg: Any,
        prompts: Any,
        engine: Any,
        environment_info: Dict[str, Any],
    ):
        self.cfg = cfg
        self.prompts = prompts
        self.engine = engine
        self.environment_info = environment_info

    # -- record assembly helpers -------------------------------------------

    def _controller_event(
        self,
        index: int,
        subtype: str,
        decision_step: Optional[int],
        message_id: Optional[str] = None,
        note: Optional[str] = None,
        recipient: str = "celab",
    ) -> Event:
        return Event(
            event_index=index,
            event_type="controller",
            decision_step=decision_step,
            speaker="controller",
            recipient=recipient,
            message_id=message_id,
            raw_output=note,
            parse_status="not_parsed",
            controller_subtype=subtype,
        )

    def _generation_event(
        self,
        index: int,
        agent: Agent,
        result: Any,
        decision_step: Optional[int],
        visible_ids: List[str],
        parsed_action: Optional[str] = None,
        parsed_body: Optional[str] = None,
        parse_status: str = "ok",
        parse_error: Optional[str] = None,
        forced_final: bool = False,
        recipient: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Event:
        return Event(
            event_index=index,
            event_type="model_generation",
            decision_step=decision_step,
            speaker=agent.name,
            recipient=recipient,
            message_id=message_id,
            raw_output=result.raw_output,
            parsed_action=parsed_action,
            parsed_body=parsed_body,
            parse_status=parse_status,
            parse_error=parse_error,
            input_tokens=result.input_tokens,
            generated_tokens=result.generated_tokens,
            generation_seed=result.generation_seed,
            finish_reason=result.finish_reason,
            generation_cap_reached=result.generation_cap_reached,
            forced_final=forced_final,
            visible_history_message_ids=list(visible_ids),
        )

    def _base_record(
        self,
        question: Dict[str, Any],
        run_index: int,
        run_seed: int,
        run_id: str,
    ) -> Dict[str, Any]:
        engine_info = self.engine.info()
        return {
            "experiment_id": self.cfg.experiment_id,
            "experiment_version": self.cfg.experiment_version,
            "architecture": self.cfg.architecture,
            "run_id": run_id,
            "dataset": self.cfg.dataset,
            "dataset_split": self.cfg.dataset_split,
            "question_id": str(question["question_id"]),
            "run_index": run_index,
            "run_seed": run_seed,
            "sample_selection_seed": self.cfg.sample_selection_seed,
            "question": question["question"],
            "gold_answer": question["answer"],
            "question_type": question["q_type"],
            "supporting_titles": list(question["supporting_titles"]),
            "private_evidence_alice": question["evidence_alice"],
            "private_evidence_bob": question["evidence_bob"],
            "alice_private_context": question["evidence_alice"],
            "bob_private_context": question["evidence_bob"],
            "dataset_metadata": dict(question.get("hotpotqa_metadata", {})),
            "supporting_facts": [
                list(fact) for fact in question.get("supporting_facts", [])
            ],
            "partition_metadata": dict(
                question.get("partition_metadata", {})
            ),
            "prompt_version": self.prompts.version,
            "prompt_hashes": dict(self.prompts.hashes),
            "model_name": self.cfg.model_name,
            "model_revision": engine_info.get("model_revision"),
            "tokenizer_revision": engine_info.get("tokenizer_revision"),
            "generation_config": {
                **self.cfg.generation.to_dict(),
                "max_decision_steps": self.cfg.max_decision_steps,
            },
            "config": self.cfg.to_dict(),
            "environment": dict(self.environment_info),
            "engine_info": engine_info,
            "events": [],
        }

    def _run_centralized_one(
        self,
        question: Dict[str, Any],
        run_index: int,
        run_seed: int,
        run_id: str,
    ) -> Dict[str, Any]:
        """Run the one-model, full-evidence diagnostic condition."""
        record = self._base_record(question, run_index, run_seed, run_id)
        events: List[Event] = []
        started = time.time()
        seed_all(run_seed)
        reader = Agent(
            "celab",
            self.prompts.render("centralized_system"),
            self.engine,
            run_seed,
        )
        task = self.prompts.render(
            "centralized_task",
            question=question["question"],
            evidence_alice=question["evidence_alice"],
            evidence_bob=question["evidence_bob"],
        )
        task_message = Message(
            message_id=reader.next_message_id(),
            speaker="controller",
            recipient="celab",
            content=task,
        )
        reader.add_message(task_message)
        events.append(
            self._controller_event(
                0,
                "centralized_task",
                decision_step=None,
                message_id=task_message.message_id,
                note="question and both evidence documents delivered to reader",
            )
        )

        final_answer: Optional[str] = None
        final_raw_output: Optional[str] = None
        error_message: Optional[str] = None
        result: Any = None
        try:
            result = reader.generate()
            final_raw_output = result.raw_output
            final_answer = result.raw_output.strip()
            message_id = reader.next_message_id()
            events.append(
                self._generation_event(
                    len(events),
                    reader,
                    result,
                    1,
                    reader.visible_message_ids(),
                    parsed_action="direct_answer",
                    parsed_body=final_answer,
                    parse_status="not_required",
                    message_id=message_id,
                )
            )
            termination_reason = "direct_answer"
        except Exception as exc:  # noqa: BLE001
            termination_reason = "error"
            error_message = str(exc)
            events.append(
                self._controller_event(
                    len(events),
                    "error_termination",
                    decision_step=1,
                    note=f"centralized reader aborted by exception: {exc}",
                )
            )

        f1, em = evaluate_answer(final_answer, question["answer"])
        input_tokens = result.input_tokens if result is not None else 0
        generated_tokens = result.generated_tokens if result is not None else 0
        generation_cap = bool(
            result is not None and result.generation_cap_reached
        )
        record.update(
            {
                "decision_steps": 1,
                "cap_reached": False,
                "decision_cap_reached": False,
                "generation_cap_reached": generation_cap,
                "forced_final_calls": 0,
                "natural_termination": termination_reason == "direct_answer",
                "termination_reason": termination_reason,
                "final_raw_output": final_raw_output,
                "final_answer": final_answer,
                "final_answer_extracted": final_answer,
                "final_parse_fallback": False,
                "protocol_parse_fallback": False,
                "parse_fallback_events": 0,
                "parse_fallback_statuses": [],
                "casefold_route_fallback": False,
                "terminal_final_precedence_fallback": False,
                "parse_status": "not_required" if result is not None else "error",
                "parse_error": None,
                "f1": f1,
                "em": em,
                "answer_f1": f1,
                "answer_em": em,
                "num_alice_queries": 0,
                "num_bob_queries": 0,
                "num_total_queries": 0,
                "num_alice_responses": 0,
                "num_bob_responses": 0,
                "num_messages": 1 if result is not None else 0,
                "num_alice_clarification_requests": 0,
                "num_bob_clarification_requests": 0,
                "num_clarification_requests": 0,
                "num_clarification_round_trips_completed": 0,
                "clarification_protocol_violations": 0,
                "ignored_clarification_requests": 0,
                "clarification_triggered": False,
                "generation_cap_agents": ["celab"] if generation_cap else [],
                "generation_cap_events": 1 if generation_cap else 0,
                "input_tokens_alice": 0,
                "input_tokens_bob": 0,
                "input_tokens_celab": input_tokens,
                "input_tokens_total": input_tokens,
                "alice_input_tokens": 0,
                "bob_input_tokens": 0,
                "celab_input_tokens": input_tokens,
                "total_input_tokens": input_tokens,
                "generated_tokens_alice": 0,
                "generated_tokens_bob": 0,
                "generated_tokens_celab": generated_tokens,
                "total_generated_tokens": generated_tokens,
                "alice_generated_tokens": 0,
                "bob_generated_tokens": 0,
                "celab_generated_tokens": generated_tokens,
                "total_model_tokens": input_tokens + generated_tokens,
                "duration_seconds": round(time.time() - started, 3),
                "error": error_message,
            }
        )
        record["events"] = [event.to_dict() for event in events]
        return record

    def _run_one_shot_gather(
        self,
        question: Dict[str, Any],
        run_index: int,
        run_seed: int,
        run_id: str,
    ) -> Dict[str, Any]:
        """Ask each private worker once, then let Celab synthesize once."""
        record = self._base_record(question, run_index, run_seed, run_id)
        events: List[Event] = []
        started = time.time()
        seed_all(run_seed)
        agents = make_agents(
            self.prompts,
            self.engine,
            run_seed,
            question["evidence_alice"],
            question["evidence_bob"],
            question=question["question"],
        )
        alice: Agent = agents["alice"]
        bob: Agent = agents["bob"]
        celab: Agent = agents["celab"]
        token_sums = {
            "alice": {"input": 0, "generated": 0},
            "bob": {"input": 0, "generated": 0},
            "celab": {"input": 0, "generated": 0},
        }
        generation_cap_agents: List[str] = []
        generation_cap_events = 0
        num_messages = 0
        num_queries = {"alice": 0, "bob": 0}
        num_responses = {"alice": 0, "bob": 0}
        final_answer: Optional[str] = None
        final_raw_output: Optional[str] = None
        parse_status = "not_parsed"
        parse_error: Optional[str] = None
        termination_reason: Optional[str] = None
        error_message: Optional[str] = None
        final_parse_fallback = False
        parse_fallback_statuses: List[str] = []

        def record_generation(
            agent: Agent,
            result: Any,
            decision_step: Optional[int],
            message_id: str,
            **kwargs: Any,
        ) -> None:
            nonlocal generation_cap_events, num_messages
            events.append(
                self._generation_event(
                    len(events),
                    agent,
                    result,
                    decision_step,
                    agent.visible_message_ids(),
                    message_id=message_id,
                    **kwargs,
                )
            )
            num_messages += 1
            token_sums[agent.name]["input"] += result.input_tokens
            token_sums[agent.name]["generated"] += result.generated_tokens
            if result.generation_cap_reached:
                generation_cap_events += 1
                if agent.name not in generation_cap_agents:
                    generation_cap_agents.append(agent.name)

        try:
            question_message = Message(
                message_id=celab.next_message_id(),
                speaker="controller",
                recipient="celab",
                content=question["question"],
            )
            celab.add_message(question_message)
            events.append(
                self._controller_event(
                    len(events),
                    "initial_task",
                    decision_step=None,
                    message_id=question_message.message_id,
                    note=f"question delivered to celab: {question['question']}",
                )
            )

            for worker in (alice, bob):
                task = self.prompts.render(
                    "worker_task", worker_name=worker.name.title()
                )
                task_message = Message(
                    message_id=worker.next_message_id(),
                    speaker="controller",
                    recipient=worker.name,
                    content=task,
                )
                worker.add_message(task_message)
                num_queries[worker.name] += 1
                events.append(
                    self._controller_event(
                        len(events),
                        "one_shot_worker_task",
                        decision_step=None,
                        message_id=task_message.message_id,
                        note=task,
                        recipient=worker.name,
                    )
                )
                worker_result = worker.generate()
                reply_id = worker.next_message_id()
                num_responses[worker.name] += 1
                record_generation(
                    worker,
                    worker_result,
                    None,
                    reply_id,
                    parse_status="not_parsed",
                    recipient="celab",
                )
                reply = Message(
                    message_id=reply_id,
                    speaker=worker.name,
                    recipient="celab",
                    content=worker_result.raw_output,
                )
                worker.add_message(reply)
                celab.add_message(reply)

            result = celab.generate()
            final_raw_output = result.raw_output
            final_id = celab.next_message_id()
            parsed = parse_celab_output(result.raw_output)
            parse_status = parsed.status
            if parsed.status == "error" or parsed.action != "final":
                parse_status = "error"
                parse_error = parsed.error or (
                    "one-shot Celab output was not a final action"
                )
                termination_reason = "parse_error"
                record_generation(
                    celab,
                    result,
                    1,
                    final_id,
                    parse_status="error",
                    parse_error=parse_error,
                )
            else:
                final_answer = parsed.body
                final_parse_fallback = (
                    parsed.status == "ok_unclosed_final_fallback"
                )
                if parsed.status.startswith("ok_") and parsed.status.endswith(
                    "_fallback"
                ):
                    parse_fallback_statuses.append(parsed.status)
                termination_reason = "natural_final"
                record_generation(
                    celab,
                    result,
                    1,
                    final_id,
                    parsed_action="final",
                    parsed_body=parsed.body,
                    parse_status=parsed.status,
                )
        except Exception as exc:  # noqa: BLE001
            termination_reason = "error"
            parse_status = "error"
            error_message = str(exc)
            events.append(
                self._controller_event(
                    len(events),
                    "error_termination",
                    decision_step=1,
                    note=f"one-shot gather aborted by exception: {exc}",
                )
            )

        f1, em = evaluate_answer(final_answer, question["answer"])
        input_total = sum(value["input"] for value in token_sums.values())
        generated_total = sum(
            value["generated"] for value in token_sums.values()
        )
        record.update(
            {
                "decision_steps": 1,
                "cap_reached": False,
                "decision_cap_reached": False,
                "generation_cap_reached": generation_cap_events > 0,
                "forced_final_calls": 0,
                "natural_termination": termination_reason == "natural_final",
                "termination_reason": termination_reason,
                "final_raw_output": final_raw_output,
                "final_answer": final_answer,
                "final_answer_extracted": final_answer,
                "final_parse_fallback": final_parse_fallback,
                "protocol_parse_fallback": bool(parse_fallback_statuses),
                "parse_fallback_events": len(parse_fallback_statuses),
                "parse_fallback_statuses": parse_fallback_statuses,
                "casefold_route_fallback": False,
                "terminal_final_precedence_fallback": (
                    "ok_terminal_final_precedence_fallback"
                    in parse_fallback_statuses
                ),
                "parse_status": parse_status,
                "parse_error": parse_error,
                "f1": f1,
                "em": em,
                "answer_f1": f1,
                "answer_em": em,
                "num_alice_queries": num_queries["alice"],
                "num_bob_queries": num_queries["bob"],
                "num_total_queries": sum(num_queries.values()),
                "num_alice_responses": num_responses["alice"],
                "num_bob_responses": num_responses["bob"],
                "num_messages": num_messages,
                "num_alice_clarification_requests": 0,
                "num_bob_clarification_requests": 0,
                "num_clarification_requests": 0,
                "num_clarification_round_trips_completed": 0,
                "clarification_protocol_violations": 0,
                "ignored_clarification_requests": 0,
                "clarification_triggered": False,
                "generation_cap_agents": generation_cap_agents,
                "generation_cap_events": generation_cap_events,
                "input_tokens_alice": token_sums["alice"]["input"],
                "input_tokens_bob": token_sums["bob"]["input"],
                "input_tokens_celab": token_sums["celab"]["input"],
                "input_tokens_total": input_total,
                "alice_input_tokens": token_sums["alice"]["input"],
                "bob_input_tokens": token_sums["bob"]["input"],
                "celab_input_tokens": token_sums["celab"]["input"],
                "total_input_tokens": input_total,
                "generated_tokens_alice": token_sums["alice"]["generated"],
                "generated_tokens_bob": token_sums["bob"]["generated"],
                "generated_tokens_celab": token_sums["celab"]["generated"],
                "total_generated_tokens": generated_total,
                "alice_generated_tokens": token_sums["alice"]["generated"],
                "bob_generated_tokens": token_sums["bob"]["generated"],
                "celab_generated_tokens": token_sums["celab"]["generated"],
                "total_model_tokens": input_total + generated_total,
                "duration_seconds": round(time.time() - started, 3),
                "error": error_message,
            }
        )
        record["events"] = [event.to_dict() for event in events]
        return record

    # -- the state machine --------------------------------------------------

    def run_one(
        self,
        question: Dict[str, Any],
        run_index: int,
        run_seed: int,
        run_id: str,
    ) -> Dict[str, Any]:
        if self.cfg.architecture == "centralized_reader":
            return self._run_centralized_one(
                question, run_index, run_seed, run_id
            )
        if self.cfg.architecture == "one_shot_gather":
            return self._run_one_shot_gather(
                question, run_index, run_seed, run_id
            )
        record = self._base_record(question, run_index, run_seed, run_id)
        events: List[Event] = []
        started = time.time()
        seed_all(run_seed)

        agents = make_agents(
            self.prompts,
            self.engine,
            run_seed,
            question["evidence_alice"],
            question["evidence_bob"],
            question=(
                question["question"]
                if self.cfg.share_question_with_workers
                else ""
            ),
        )
        alice: Agent = agents["alice"]
        bob: Agent = agents["bob"]
        celab: Agent = agents["celab"]

        # Aggregated counters, filled in as the run proceeds.
        counters: Dict[str, Any] = {
            "num_alice_queries": 0,
            "num_bob_queries": 0,
            "num_alice_responses": 0,
            "num_bob_responses": 0,
            "num_messages": 0,
            "forced_final_calls": 0,
            "generation_cap_agents": [],
            "generation_cap_events": 0,
            "parse_fallback_events": 0,
            "parse_fallback_statuses": [],
            "num_alice_clarification_requests": 0,
            "num_bob_clarification_requests": 0,
            "num_clarification_round_trips_completed": 0,
            "clarification_protocol_violations": 0,
            "ignored_clarification_requests": 0,
        }
        token_sums: Dict[str, int] = {
            "input_tokens_alice": 0,
            "input_tokens_bob": 0,
            "input_tokens_celab": 0,
            "generated_tokens_alice": 0,
            "generated_tokens_bob": 0,
            "generated_tokens_celab": 0,
        }
        decision_steps = 0
        natural_termination = False
        termination_reason: Optional[str] = None
        final_answer: Optional[str] = None
        final_raw_output: Optional[str] = None
        final_parse_fallback = False
        parse_status = "not_parsed"
        parse_error: Optional[str] = None
        error_message: Optional[str] = None
        clarification_counts = {"alice": 0, "bob": 0}
        expected_clarification_target: Optional[str] = None

        def add_event(event: Event) -> None:
            event.event_index = len(events)
            events.append(event)

        def record_generation(
            agent: Agent,
            result: Any,
            decision_step: Optional[int],
            message_id: Optional[str],
            **kwargs: Any,
        ) -> None:
            add_event(
                self._generation_event(
                    len(events),
                    agent,
                    result,
                    decision_step,
                    agent.visible_message_ids(),
                    message_id=message_id,
                    **kwargs,
                )
            )
            counters["num_messages"] += 1
            token_sums[f"input_tokens_{agent.name}"] += result.input_tokens
            token_sums[f"generated_tokens_{agent.name}"] += result.generated_tokens
            if result.generation_cap_reached:
                counters["generation_cap_events"] += 1
                if agent.name not in counters["generation_cap_agents"]:
                    counters["generation_cap_agents"].append(agent.name)

        def record_parse_fallback(status: str) -> None:
            if status.startswith("ok_") and status.endswith("_fallback"):
                counters["parse_fallback_events"] += 1
                if status not in counters["parse_fallback_statuses"]:
                    counters["parse_fallback_statuses"].append(status)

        try:
            # Initial task: the question is delivered to Celab as its first
            # user turn (spec sec. 5.3; Gemma-3 template constraint).
            task_message = Message(
                message_id=celab.next_message_id(),
                speaker="controller",
                recipient="celab",
                content=question["question"],
            )
            celab.add_message(task_message)
            add_event(
                self._controller_event(
                    len(events),
                    "initial_task",
                    decision_step=None,
                    message_id=task_message.message_id,
                    note=f"question delivered to celab: {question['question']}",
                )
            )

            while decision_steps < self.cfg.max_decision_steps:
                decision_steps += 1
                result = celab.generate()
                celab_message_id = celab.next_message_id()
                parsed = parse_celab_output(result.raw_output)

                # A clarification request gives Celab one opportunity to
                # respond to the same worker. This is observed and logged,
                # not enforced by rewriting or retrying the model output.
                if expected_clarification_target is not None:
                    expected_action = f"ask_{expected_clarification_target}"
                    if parsed.action == expected_action:
                        counters[
                            "num_clarification_round_trips_completed"
                        ] += 1
                    else:
                        counters["clarification_protocol_violations"] += 1
                    expected_clarification_target = None

                if parsed.status == "error":
                    # Parse error: keep raw output, end the run (spec sec. 7).
                    parse_status = "error"
                    parse_error = parsed.error
                    termination_reason = "parse_error"
                    final_raw_output = result.raw_output
                    record_generation(
                        celab,
                        result,
                        decision_steps,
                        celab_message_id,
                        parse_status="error",
                        parse_error=parsed.error,
                    )
                    add_event(
                        self._controller_event(
                            len(events),
                            "parse_error_termination",
                            decision_step=decision_steps,
                            note=f"run terminated: {parsed.error}",
                        )
                    )
                    break

                record_parse_fallback(parsed.status)

                if parsed.action == "final":
                    parse_status = parsed.status
                    final_parse_fallback = (
                        parsed.status == "ok_unclosed_final_fallback"
                    )
                    final_answer = parsed.body
                    final_raw_output = result.raw_output
                    natural_termination = True
                    termination_reason = "natural_final"
                    record_generation(
                        celab,
                        result,
                        decision_steps,
                        celab_message_id,
                        parsed_action="final",
                        parsed_body=parsed.body,
                        parse_status=parsed.status,
                    )
                    break

                # ask_alice / ask_bob: relay the request through the link.
                worker_name = (
                    "alice" if parsed.action == "ask_alice" else "bob"
                )
                worker = alice if worker_name == "alice" else bob
                counters[f"num_{worker_name}_queries"] += 1
                request_message = Message(
                    message_id=celab_message_id,
                    speaker="celab",
                    recipient=worker_name,
                    content=result.raw_output,
                )
                record_generation(
                    celab,
                    result,
                    decision_steps,
                    celab_message_id,
                    parsed_action=parsed.action,
                    parse_status=parsed.status,
                    recipient=worker_name,
                )
                # Deliver the request, then run the worker. The same Message
                # object lives in both histories (explicit communication).
                worker.add_message(request_message)
                celab.add_message(request_message)
                worker_result = worker.generate()
                worker_message_id = worker.next_message_id()
                counters[f"num_{worker_name}_responses"] += 1
                worker_parsed_action = None
                worker_parsed_body = None
                worker_parse_status = "not_parsed"
                if self.cfg.worker_clarification.enabled:
                    worker_parsed = parse_worker_reply(worker_result.raw_output)
                    if worker_parsed.action == "clarify":
                        if (
                            clarification_counts[worker_name]
                            < self.cfg.worker_clarification.max_per_worker
                        ):
                            clarification_counts[worker_name] += 1
                            counters[
                                f"num_{worker_name}_clarification_requests"
                            ] += 1
                            expected_clarification_target = worker_name
                            worker_parsed_action = "clarify"
                            worker_parsed_body = worker_parsed.body
                            worker_parse_status = "worker_clarification"
                        else:
                            counters["ignored_clarification_requests"] += 1
                            worker_parse_status = (
                                "worker_clarification_ignored_limit"
                            )
                record_generation(
                    worker,
                    worker_result,
                    decision_steps,
                    worker_message_id,
                    parsed_action=worker_parsed_action,
                    parsed_body=worker_parsed_body,
                    parse_status=worker_parse_status,
                    recipient="celab",
                )
                reply_message = Message(
                    message_id=worker_message_id,
                    speaker=worker_name,
                    recipient="celab",
                    content=worker_result.raw_output,
                )
                worker.add_message(reply_message)
                celab.add_message(reply_message)

            # Decision cap: only when the loop ran to the limit with no
            # termination and no parse error.
            if (
                decision_steps >= self.cfg.max_decision_steps
                and termination_reason is None
            ):
                if expected_clarification_target is not None:
                    counters["clarification_protocol_violations"] += 1
                    expected_clarification_target = None
                counters["forced_final_calls"] = 1
                add_event(
                    self._controller_event(
                        len(events),
                        "decision_cap_reached",
                        decision_step=decision_steps,
                        note=(
                            f"max_decision_steps={self.cfg.max_decision_steps} "
                            "reached without a final answer"
                        ),
                    )
                )
                instruction = self.prompts.render("forced_final")
                instruction_message = Message(
                    message_id=celab.next_message_id(),
                    speaker="controller",
                    recipient="celab",
                    content=instruction,
                )
                celab.add_message(instruction_message)
                add_event(
                    self._controller_event(
                        len(events),
                        "forced_final_instruction",
                        decision_step=decision_steps,
                        message_id=instruction_message.message_id,
                        note=instruction,
                    )
                )
                # Forced-final call: keeps decision_steps at the cap (no
                # step 21); its generated tokens ARE counted (spec sec. 8).
                result = celab.generate()
                final_raw_output = result.raw_output
                celab_message_id = celab.next_message_id()
                forced_parse = extract_final_answer(result.raw_output)
                record_parse_fallback(forced_parse.status)
                if forced_parse.status != "error":
                    final_answer = forced_parse.body
                    termination_reason = "forced_final"
                    parse_status = forced_parse.status
                    final_parse_fallback = (
                        forced_parse.status == "ok_unclosed_final_fallback"
                    )
                else:
                    termination_reason = "forced_final_parse_failure"
                    parse_status = "error"
                    parse_error = forced_parse.error
                record_generation(
                    celab,
                    result,
                    decision_steps,
                    celab_message_id,
                    parsed_action=(
                        "final" if forced_parse.status != "error" else None
                    ),
                    parsed_body=(
                        forced_parse.body
                        if forced_parse.status != "error"
                        else None
                    ),
                    parse_status=forced_parse.status,
                    parse_error=forced_parse.error,
                    forced_final=True,
                )
        except Exception as exc:  # noqa: BLE001 - log any failure, keep partial record
            termination_reason = "error"
            parse_status = "error"
            error_message = str(exc)
            add_event(
                self._controller_event(
                    len(events),
                    "error_termination",
                    decision_step=decision_steps,
                    note=f"run aborted by exception: {exc}",
                )
            )

        # Evaluation: null/empty predictions go through the official
        # evaluator as empty strings and score 0 (plan decision 13).
        f1, em = evaluate_answer(final_answer, question["answer"])

        input_tokens_total = sum(
            token_sums[f"input_tokens_{name}"] for name in ("alice", "bob", "celab")
        )
        generated_tokens_total = sum(
            token_sums[f"generated_tokens_{name}"]
            for name in ("alice", "bob", "celab")
        )
        record.update(
            {
                "decision_steps": decision_steps,
                # The decision cap is "reached" only when the loop ran to the
                # limit without terminating, i.e. the forced final was
                # triggered (a natural final on step 20 is not a cap hit).
                "cap_reached": counters["forced_final_calls"] > 0,
                "decision_cap_reached": counters["forced_final_calls"] > 0,
                "generation_cap_reached": counters["generation_cap_events"] > 0,
                "forced_final_calls": counters["forced_final_calls"],
                "natural_termination": natural_termination,
                "termination_reason": termination_reason,
                "final_raw_output": final_raw_output,
                "final_answer": final_answer,
                "final_answer_extracted": final_answer,
                "final_parse_fallback": final_parse_fallback,
                "protocol_parse_fallback": bool(
                    counters["parse_fallback_events"]
                ),
                "parse_fallback_events": counters["parse_fallback_events"],
                "parse_fallback_statuses": counters[
                    "parse_fallback_statuses"
                ],
                "casefold_route_fallback": (
                    "ok_casefold_route_fallback"
                    in counters["parse_fallback_statuses"]
                ),
                "terminal_final_precedence_fallback": (
                    "ok_terminal_final_precedence_fallback"
                    in counters["parse_fallback_statuses"]
                ),
                "parse_status": parse_status,
                "parse_error": parse_error,
                "f1": f1,
                "em": em,
                "answer_f1": f1,
                "answer_em": em,
                "num_alice_queries": counters["num_alice_queries"],
                "num_bob_queries": counters["num_bob_queries"],
                "num_total_queries": (
                    counters["num_alice_queries"] + counters["num_bob_queries"]
                ),
                "num_alice_responses": counters["num_alice_responses"],
                "num_bob_responses": counters["num_bob_responses"],
                "num_messages": counters["num_messages"],
                "num_alice_clarification_requests": counters[
                    "num_alice_clarification_requests"
                ],
                "num_bob_clarification_requests": counters[
                    "num_bob_clarification_requests"
                ],
                "num_clarification_requests": (
                    counters["num_alice_clarification_requests"]
                    + counters["num_bob_clarification_requests"]
                ),
                "num_clarification_round_trips_completed": counters[
                    "num_clarification_round_trips_completed"
                ],
                "clarification_protocol_violations": counters[
                    "clarification_protocol_violations"
                ],
                "ignored_clarification_requests": counters[
                    "ignored_clarification_requests"
                ],
                "clarification_triggered": bool(
                    counters["num_alice_clarification_requests"]
                    + counters["num_bob_clarification_requests"]
                ),
                "generation_cap_agents": counters["generation_cap_agents"],
                "generation_cap_events": counters["generation_cap_events"],
                "input_tokens_alice": token_sums["input_tokens_alice"],
                "input_tokens_bob": token_sums["input_tokens_bob"],
                "input_tokens_celab": token_sums["input_tokens_celab"],
                "input_tokens_total": input_tokens_total,
                "alice_input_tokens": token_sums["input_tokens_alice"],
                "bob_input_tokens": token_sums["input_tokens_bob"],
                "celab_input_tokens": token_sums["input_tokens_celab"],
                "total_input_tokens": input_tokens_total,
                "generated_tokens_alice": token_sums["generated_tokens_alice"],
                "generated_tokens_bob": token_sums["generated_tokens_bob"],
                "generated_tokens_celab": token_sums["generated_tokens_celab"],
                "total_generated_tokens": generated_tokens_total,
                "alice_generated_tokens": token_sums["generated_tokens_alice"],
                "bob_generated_tokens": token_sums["generated_tokens_bob"],
                "celab_generated_tokens": token_sums["generated_tokens_celab"],
                # Auxiliary combined count (input + generated); the primary
                # efficiency metric is total_generated_tokens (spec sec. 12).
                "total_model_tokens": input_tokens_total + generated_tokens_total,
                "duration_seconds": round(time.time() - started, 3),
                "error": error_message,
            }
        )
        record["events"] = [event.to_dict() for event in events]
        return record

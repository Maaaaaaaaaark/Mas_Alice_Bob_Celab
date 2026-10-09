"""Prompt building for training: roles, evidence, and label hygiene."""

from __future__ import annotations

from hotpot_mas.training.prompts_builder import (
    TRAINING_PROMPT_NAMES,
    TrainingPrompts,
    WORKER_DISPLAY_NAMES,
)
from tests.training_test_utils import (
    GOLD_ANSWER,
    PROMPTS_TRAINING_DIR,
    make_rows,
    select_balanced_questions,
)


class TestWorkerMessages:
    def test_alice_and_bob_use_distinct_system_prompts(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        a = prompts.worker_messages("A", "Question?", "evidence text")
        b = prompts.worker_messages("B", "Question?", "evidence text")
        assert a[0]["content"] != b[0]["content"]
        assert "Alice" in a[0]["content"]
        assert "Bob" in b[0]["content"]

    def test_question_and_private_evidence_are_substituted(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        messages = prompts.worker_messages(
            "A", "What is the capital?", "Title: X\nsecret fact"
        )
        joined = " ".join(m["content"] for m in messages)
        assert "What is the capital?" in joined
        assert "secret fact" in joined

    def test_worker_task_names_the_worker(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        messages = prompts.worker_messages("B", "Question?", "evidence")
        assert "Bob" in messages[-1]["content"]
        assert WORKER_DISPLAY_NAMES["B"] == "Bob"

    def test_messages_have_system_and_user_roles(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        messages = prompts.worker_messages("A", "Question?", "evidence")
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_unknown_worker_rejected(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        try:
            prompts.worker_messages("C", "Question?", "evidence")
        except ValueError as exc:
            assert "unknown worker" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestSynthesizerMessages:
    def test_contains_question_and_both_reports(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        messages = prompts.synthesizer_messages(
            "Original question?", "report from alice", "report from bob"
        )
        joined = " ".join(m["content"] for m in messages)
        assert "Original question?" in joined
        assert "report from alice" in joined
        assert "report from bob" in joined

    def test_gold_answer_never_included(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        messages = prompts.synthesizer_messages(
            "Original question?", "report a", "report b"
        )
        joined = " ".join(m["content"] for m in messages)
        assert GOLD_ANSWER not in joined


class TestPromptSetHygiene:
    def test_no_supporting_labels_anywhere_in_templates(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        for name in TRAINING_PROMPT_NAMES:
            text = prompts.prompt_set.templates[name].lower()
            assert "supporting" not in text
            assert "distractor" not in text

    def test_rendered_worker_input_has_no_labels(self):
        question = select_balanced_questions(
            make_rows(1), 1, selection_seed=0
        )[0]
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        for worker in ("A", "B"):
            evidence = (
                question.evidence_alice
                if worker == "A"
                else question.evidence_bob
            )
            messages = prompts.worker_messages(
                worker, question.question, evidence
            )
            joined = " ".join(m["content"] for m in messages).lower()
            assert "supporting" not in joined
            assert "is_supporting" not in joined

    def test_prompt_version_is_present(self):
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        assert prompts.version.startswith("v1-")
        assert set(prompts.hashes) == set(TRAINING_PROMPT_NAMES)

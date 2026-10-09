"""Single-question rollout: G*G C calls, reward matrix, signal filtering,
zero-length-report exclusion, seed reproducibility."""

from __future__ import annotations

import pytest

from hotpot_mas.training.config import WorkerTrainingConfig
from hotpot_mas.training.fake_policy import FakePolicy
from hotpot_mas.training.prompts_builder import TrainingPrompts
from hotpot_mas.training.rollout import rollout_question
from tests.training_test_utils import (
    GOLD_ANSWER,
    MATRIX_A_ONLY,
    MATRIX_BOTH,
    MATRIX_B_ONLY,
    MATRIX_NONE,
    PROMPTS_TRAINING_DIR,
    make_rows,
    scripted_synthesizer,
    select_balanced_questions,
)


@pytest.fixture
def question():
    return select_balanced_questions(make_rows(1), 1, selection_seed=0)[0]


@pytest.fixture
def prompts() -> TrainingPrompts:
    return TrainingPrompts(PROMPTS_TRAINING_DIR)


@pytest.fixture
def workers() -> WorkerTrainingConfig:
    return WorkerTrainingConfig(
        G=2,
        delta=0.05,
        eps_n=1e-6,
        questions_per_update=1,
        steps=1,
        max_sampling_attempts=10,
    )


def run_rollout(question, prompts, workers, scripts, policy=None):
    synth, calls = scripted_synthesizer(scripts, workers.G)
    rollout = rollout_question(
        question,
        policy or FakePolicy(),
        synth,
        prompts,
        workers,
        base_seed=0,
    )
    return rollout, synth, calls


class TestRollout:
    def test_c_called_exactly_g_squared_times(self, question, prompts, workers):
        rollout, synth, calls = run_rollout(
            question, prompts, workers, {question.question: MATRIX_NONE}
        )
        assert synth.call_count == workers.G * workers.G
        assert len(calls) == 4
        # Call order is i-major: (0,0), (0,1), (1,0), (1,1).
        assert calls[0][0] == question.question
        assert [p.i for row in rollout.pairs for p in row] == [0, 0, 1, 1]
        assert [p.j for row in rollout.pairs for p in row] == [0, 1, 0, 1]

    def test_reward_matrix_from_scripted_answers(
        self, question, prompts, workers
    ):
        rollout, _, _ = run_rollout(
            question, prompts, workers, {question.question: MATRIX_A_ONLY}
        )
        assert rollout.reward_matrix() == pytest.approx(
            [[1.0, 1.0], [0.0, 0.0]]
        )

    def test_a_only_signal_keeps_only_a(self, question, prompts, workers):
        rollout, _, _ = run_rollout(
            question, prompts, workers, {question.question: MATRIX_A_ONLY}
        )
        assert rollout.signal.kept_sides == ["A"]
        assert len(rollout.kept_reports) == workers.G
        assert all(rr.side == "A" for rr in rollout.kept_reports)
        assert all(rr.advantage is not None for rr in rollout.kept_reports)

    def test_b_only_signal_keeps_only_b(self, question, prompts, workers):
        rollout, _, _ = run_rollout(
            question, prompts, workers, {question.question: MATRIX_B_ONLY}
        )
        assert rollout.signal.kept_sides == ["B"]
        assert all(rr.side == "B" for rr in rollout.kept_reports)

    def test_both_signal_keeps_both(self, question, prompts, workers):
        rollout, _, _ = run_rollout(
            question, prompts, workers, {question.question: MATRIX_BOTH}
        )
        assert rollout.signal.kept_sides == ["A", "B"]
        assert len(rollout.kept_reports) == 2 * workers.G

    def test_no_signal_question_has_no_signal(
        self, question, prompts, workers
    ):
        rollout, _, _ = run_rollout(
            question, prompts, workers, {question.question: MATRIX_NONE}
        )
        assert rollout.has_signal is False
        assert rollout.kept_reports == []

    def test_advantages_match_manual_computation(
        self, question, prompts, workers
    ):
        rollout, _, _ = run_rollout(
            question, prompts, workers, {question.question: MATRIX_A_ONLY}
        )
        # Q_A = [1, 0] -> adv = (Q - 0.5) / (0.5 + eps)
        expected = [0.999998, -0.999998]
        got = [
            rr.advantage
            for rr in sorted(rollout.kept_reports, key=lambda r: r.index)
        ]
        assert got == pytest.approx(expected, abs=1e-4)

    def test_seed_reproducibility(self, question, prompts, workers):
        scripts = {question.question: MATRIX_BOTH}
        first, _, _ = run_rollout(question, prompts, workers, scripts)
        second, _, _ = run_rollout(question, prompts, workers, scripts)
        assert (
            [r.report.token_ids for r in first.a_reports]
            == [r.report.token_ids for r in second.a_reports]
        )
        assert (
            [r.report.token_ids for r in first.b_reports]
            == [r.report.token_ids for r in second.b_reports]
        )
        assert first.reward_matrix() == second.reward_matrix()
        assert first.signal.kept_sides == second.signal.kept_sides

    def test_gold_answer_never_enters_inputs(self, question, prompts, workers):
        rollout, _, calls = run_rollout(
            question, prompts, workers, {question.question: MATRIX_NONE}
        )
        assert question.answer == GOLD_ANSWER
        for rr in rollout.a_reports + rollout.b_reports:
            for message in rr.messages:
                assert GOLD_ANSWER not in message["content"]
        for row in rollout.pairs:
            for pair in row:
                for message in pair.messages:
                    assert GOLD_ANSWER not in message["content"]


class TestEmptyReportExclusion:
    class EmptyFakePolicy(FakePolicy):
        """Fake policy whose first sampled report is always empty."""

        def __init__(self):
            super().__init__()
            self.sample_count = 0

        def sample_report(self, messages, generation_seed, decode):
            self.sample_count += 1
            if self.sample_count == 1:
                from hotpot_mas.training.policy import Report

                return Report(token_ids=[], text="", logprobs=[])
            return super().sample_report(messages, generation_seed, decode)

    def test_empty_report_excluded_from_kept_but_scored(
        self, question, prompts, workers
    ):
        policy = self.EmptyFakePolicy()
        rollout, _, _ = run_rollout(
            question,
            prompts,
            workers,
            {question.question: MATRIX_A_ONLY},
            policy=policy,
        )
        # The empty A[0] report is still scored by C (G*G calls).
        assert rollout.signal.kept_sides == ["A"]
        assert len(rollout.excluded_empty_reports) == 1
        assert rollout.excluded_empty_reports[0]["reason"] == "empty_report"
        assert all(rr.report.token_ids for rr in rollout.kept_reports)

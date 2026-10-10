"""Chat-message construction for the training experiments.

Builds the actual ``[{"role": ...}, ...]`` lists fed to the workers and the
synthesizer from the versioned templates in the training prompt directory.
The rendered worker evidence comes straight from the manifest's
``evidence_alice`` / ``evidence_bob`` strings (title + paragraph blocks,
no supporting/distractor labels anywhere); supporting-document labels
exist only in the manifest metadata and the trace, never in model inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from hotpot_mas.prompts import PromptSet

TRAINING_PROMPT_NAMES = (
    "worker_a_system",
    "worker_b_system",
    "worker_task",
    "synthesizer_system",
)

WORKER_DISPLAY_NAMES = {"A": "Alice", "B": "Bob"}


class TrainingPrompts:
    """Prompt set for the two training stages (workers + synthesizer)."""

    def __init__(self, prompt_dir: Path):
        self.prompt_set = PromptSet(prompt_dir, TRAINING_PROMPT_NAMES)
        self.version = self.prompt_set.version
        self.hashes = dict(self.prompt_set.hashes)
        self.prompt_dir = Path(prompt_dir)

    def worker_messages(
        self, worker: str, question: str, private_evidence: str
    ) -> List[Dict[str, str]]:
        """Messages for one worker: role system prompt + task instruction.

        ``worker`` is ``"A"`` or ``"B"``; ``private_evidence`` is the
        label-free evidence block from the manifest.
        """
        if worker not in WORKER_DISPLAY_NAMES:
            raise ValueError(f"unknown worker {worker!r}")
        system_name = (
            "worker_a_system" if worker == "A" else "worker_b_system"
        )
        system = self.prompt_set.render(
            system_name,
            question=question,
            private_evidence=private_evidence,
        )
        task = self.prompt_set.render(
            "worker_task", worker_name=WORKER_DISPLAY_NAMES[worker]
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

    def synthesizer_messages(
        self, question: str, a_report: str, b_report: str
    ) -> List[Dict[str, str]]:
        """Messages for C: question + Alice's report + Bob's report.

        The gold answer is never included here.
        """
        system = self.prompt_set.render("synthesizer_system")
        user = (
            f"Original question:\n{question}\n\n"
            f"Report from Alice:\n{a_report}\n\n"
            f"Report from Bob:\n{b_report}"
        )
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Original question:\nIs the stated claim true?\n\n"
                    "Report from Alice:\nThe evidence explicitly confirms the claim.\n\n"
                    "Report from Bob:\nNo conflicting information is reported."
                ),
            },
            {"role": "assistant", "content": "Answer: yes"},
            {
                "role": "user",
                "content": (
                    "Original question:\nIn what year did the event occur?\n\n"
                    "Report from Alice:\nThe event occurred in 1969.\n\n"
                    "Report from Bob:\nThe report discusses the same event."
                ),
            },
            {"role": "assistant", "content": "Answer: 1969"},
            {
                "role": "user",
                "content": (
                    "Original question:\nIs the object made of wood?\n\n"
                    "Report from Alice:\nIt is described as being made of metal.\n\n"
                    "Report from Bob:\nNo source describes it as wooden."
                ),
            },
            {"role": "assistant", "content": "Answer: no"},
            {"role": "user", "content": user},
        ]

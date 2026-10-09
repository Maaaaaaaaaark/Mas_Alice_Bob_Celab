"""Cross-paired GRPO training experiments (``cross_paired_grpo.tex``).

Two independently runnable stages:

- Stage 1 (``worker_trainer.py``): train the shared-LoRA workers A/B with a
  cross-paired GRPO objective against the frozen, greedy synthesizer C.
- Stage 2 (``synthesizer_trainer.py``): freeze the trained workers and
  supervised-fine-tune C (own LoRA) on the gold answer only, including the
  empty-report control comparison.

This package is additive: the inference baseline (``hotpot_mas`` modules,
configs, prompts) is unchanged and keeps its own CLI.
"""

"""Stage 1a: independent worker SFT.

This subpackage implements the new worker-SFT experiment specified in
``docs/WORKER_SFT.md``. One shared worker model is trained with SFT on a
local question plus one or more visible documents and must answer with
verbatim source sentences (``D0: "..."``) or ``NOT FOUND``.

It is separate from the older Stage 1 cross-paired GRPO worker training
and the Stage 2 synthesizer SFT; nothing in this package loads their
adapters, configs, or prompts by default. The historical experiments and
their saved outputs are untouched.
"""

from .config import WorkerSFTConfig

__all__ = ["WorkerSFTConfig"]

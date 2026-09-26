"""Deterministic seeding (spec sec. 10.2).

Every run re-seeds Python, NumPy, and PyTorch (CPU + CUDA) from ``run_seed``.
The engine additionally re-seeds PyTorch right before every generation call
with the same run seed, so sampled output does not depend on the number or
order of earlier calls.

Reproducibility boundary (documented in the README): identical results are
expected for the same GPU/driver/library versions. Cross-version or
cross-device bitwise reproduction is not guaranteed because some CUDA
kernels are not bitwise deterministic across versions.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) global generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_torch(seed: int) -> None:
    """Re-seed only the PyTorch generators (used before each generation call)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
